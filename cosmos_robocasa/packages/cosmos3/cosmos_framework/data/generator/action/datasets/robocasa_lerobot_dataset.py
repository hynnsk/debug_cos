# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""RoboCasa (PandaOmron / PandaMobile) LeRobot action-policy dataset.

Reads the RoboCasa ``PhysicalAI-Robotics-Manipulation-Kitchen-Demos`` LeRobot
export (codebase_version v2.1) through the official ``lerobot`` reader (via
:class:`BaseActionLeRobotDataset`), so the v2.1 per-episode layout is handled by
the library rather than a hand-rolled parquet reader.

cosmos_robocasa: the same reader serves the ORIGINAL RoboCasa benchmark (24 atomic tasks x ~50 human demos, the
Cosmos Policy export ``nvidia/RoboCasa-Cosmos-Policy``) once
``cosmos_framework/scripts/convert_robocasa_cosmos_policy_to_lerobot_v30.py`` has written it as LeRobot v3 shards
with the SAME 12D action / 16D state column layout (base channels zero-filled, control_mode = -1). That recipe uses
the 10D arm-only contract (``use_base_action=False``), ``action_normalization="quantile_rot"`` with the bundled
``robocasa_cosmos_policy_*`` stats, the ``use_state`` proprioception token normalized by its OWN statistics
(``state_normalization``), and the Cosmos Policy "stronger" image augmentation (``augment_rotation_deg`` /
``augment_hue``). See docs/action_policy_robocasa_posttrain.md.

RoboCasa's native 12D action is
``[base_motion(4), control_mode(1), eef_pos(3), eef_rot_axisangle(3), gripper(1)]``
(see ``meta/modality.json``).

With ``use_base_action=False`` the base channels are dropped and only the arm delta is kept,
re-encoding rotation axis-angle -> rot6d. That 10D contract suits task sets whose base never
moves; it reduces the embodiment to a stationary Franka arm, i.e. the LIBERO/DROID regime:

    12D  ->  slice [5:8]+[8:11]+[11]  ->  [pos(3), rot_axisangle(3), gripper(1)]  (7D)
         ->  axisangle->rot6d         ->  [pos(3), rot6d(6), gripper(1)]          (10D)

The stored ``eef_pos``/``eef_rot`` are already per-frame OSC deltas, so this is a
``frame_wise_relative`` (``backward_framewise``) action — identical semantics to
``LIBEROLeRobotDataset._build_frame_wise_action``, only the slice indices differ.

Cameras: RoboCasa exports 3 views (``robot0_eye_in_hand`` wrist + ``agentview_left``
/ ``agentview_right`` third-person). ``concat_view`` tiles them wrist-on-top,
left/right-on-bottom (mirrors ``DROIDLeRobotDataset._compose_multi_view``).
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any, ClassVar

import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T

from cosmos_framework.data.generator.action.datasets.cosmos3_action_lerobot import (
    ActionNormalization,
    ActionSpec,
    BaseActionLeRobotDataset,
    Gripper,
    Pos,
    Rot,
    build_action_spec,
)
from cosmos_framework.data.generator.action.utils.action_processing import (
    ActionNormalizer,
    load_action_stats,
    resolve_action_normalization,
)
from cosmos_framework.data.generator.action.utils.pose_utils import PoseConvention, convert_rotation
from cosmos_framework.data.generator.action.utils.video_decode_env import ensure_video_decoder_libs
from cosmos_framework.data.generator.action.utils.viewpoint_utils import Viewpoint
from cosmos_framework.utils import log

# LeRobot column names (constant across all RoboCasa atomic tasks; see meta/info.json).
_ACTION_FEATURE = "action"
_STATE_FEATURE = "observation.state"
_IMAGE_FEATURES = {
    "wrist": "observation.images.robot0_eye_in_hand",
    "left": "observation.images.robot0_agentview_left",
    "right": "observation.images.robot0_agentview_right",
}

# Native 12D action layout (meta/modality.json): the arm-only contract keeps only
# the arm end-effector delta + gripper.
_EEF_POS = slice(5, 8)  # end_effector_position delta
_EEF_ROT = slice(8, 11)  # end_effector_rotation delta (axis-angle)
_GRIPPER = slice(11, 12)  # gripper_close

# observation.state 16D layout (meta/modality.json): base_position[0:3],
# base_rotation[3:7] (quat), end_effector_position_relative[7:10] (base frame, meters),
# end_effector_rotation_relative[10:14] (quat, xyzw — verified: base_rot=[0,0,.707,.707]
# is a 90 deg yaw about z, w last), gripper_qpos[14:16] (two finger joint positions).
# For the EEF proprioception token we keep only the arm eef pose + a 1D gripper opening.
_STATE_EEF_POS = slice(7, 10)  # absolute eef position (base frame)
_STATE_EEF_ROT = slice(10, 14)  # absolute eef rotation (quaternion, xyzw)
_STATE_GRIPPER = slice(14, 16)  # gripper qpos (two fingers)


# All 18 ``target/atomic`` tasks. Base-motion prevalence measured over 40 episodes each:
# NavigateKitchen is the only genuinely mobile task (100% of episodes, 2.6 m mean net
# displacement); PickPlaceDrawerToCounter moves in 48% (<=0.71 m); five more move in
# 2-10% of episodes (<=0.4 m); the remaining ten never move the base (<6 mm noise).
# Training on this set requires ``use_base_action=True`` so the base channels exist.
DEFAULT_ALL_ATOMIC_TASKS: tuple[str, ...] = (
    "CloseBlenderLid",
    "CloseFridge",
    "CloseToasterOvenDoor",
    "CoffeeSetupMug",
    "NavigateKitchen",
    "OpenCabinet",
    "OpenDrawer",
    "OpenStandMixerHead",
    "PickPlaceCounterToCabinet",
    "PickPlaceCounterToStove",
    "PickPlaceDrawerToCounter",
    "PickPlaceSinkToCounter",
    "PickPlaceToasterToCounter",
    "SlideDishwasherRack",
    "TurnOffStove",
    "TurnOnElectricKettle",
    "TurnOnMicrowave",
    "TurnOnSinkFaucet",
)


# cosmos_robocasa: the 24 atomic tasks of the ORIGINAL RoboCasa benchmark (RSS 2024; Cosmos Policy's RoboCasa
# setting, ``nvidia/RoboCasa-Cosmos-Policy/success_only``). All fixed-base (the mobile base never moves), so the
# 10D arm-only contract (``use_base_action=False``) covers them. Shards are discovered as ``<root>/<Task>/<date>/lerobot``.
DEFAULT_ROBOCASA_COSMOS_POLICY_TASKS: tuple[str, ...] = (
    "PnPCounterToCab",
    "PnPCabToCounter",
    "PnPCounterToSink",
    "PnPSinkToCounter",
    "PnPCounterToMicrowave",
    "PnPMicrowaveToCounter",
    "PnPCounterToStove",
    "PnPStoveToCounter",
    "OpenSingleDoor",
    "CloseSingleDoor",
    "OpenDoubleDoor",
    "CloseDoubleDoor",
    "OpenDrawer",
    "CloseDrawer",
    "TurnOnSinkFaucet",
    "TurnOffSinkFaucet",
    "TurnSinkSpout",
    "TurnOnStove",
    "TurnOffStove",
    "CoffeeSetupMug",
    "CoffeeServeMug",
    "CoffeePressButton",
    "TurnOnMicrowave",
    "TurnOffMicrowave",
)


# The 65 ``pretrain/atomic`` tasks (every atomic task RoboCasa365 ships a pretrain split for;
# the 18 ``target/atomic`` names above are a subset). Human demos only -- the registry also
# lists machine-generated ``mg`` paths for 60 of these, but they are not part of the local
# download. Same 12D action / 16D state / 3-camera contract as target/atomic, so this loader
# handles them unchanged; only ``root`` and ``task_names`` differ.
DEFAULT_PRETRAIN_ATOMIC_TASKS: tuple[str, ...] = (
    "AdjustToasterOvenTemperature", "AdjustWaterTemperature", "CheesyBread", "CloseBlenderLid",
    "CloseCabinet", "CloseDishwasher", "CloseDrawer", "CloseElectricKettleLid", "CloseFridge",
    "CloseFridgeDrawer", "CloseMicrowave", "CloseOven", "CloseStandMixerHead",
    "CloseToasterOvenDoor", "CoffeeServeMug", "CoffeeSetupMug", "LowerHeat", "MakeIcedCoffee",
    "NavigateKitchen", "OpenBlenderLid", "OpenCabinet", "OpenDishwasher", "OpenDrawer",
    "OpenElectricKettleLid", "OpenFridge", "OpenFridgeDrawer", "OpenMicrowave", "OpenOven",
    "OpenStandMixerHead", "OpenToasterOvenDoor", "PackDessert", "PickPlaceCabinetToCounter",
    "PickPlaceCounterToBlender", "PickPlaceCounterToCabinet", "PickPlaceCounterToDrawer",
    "PickPlaceCounterToMicrowave", "PickPlaceCounterToOven", "PickPlaceCounterToSink",
    "PickPlaceCounterToStandMixer", "PickPlaceCounterToStove", "PickPlaceCounterToToasterOven",
    "PickPlaceDrawerToCounter", "PickPlaceFridgeDrawerToShelf", "PickPlaceFridgeShelfToDrawer",
    "PickPlaceMicrowaveToCounter", "PickPlaceSinkToCounter", "PickPlaceStoveToCounter",
    "PickPlaceToasterOvenToCounter", "PickPlaceToasterToCounter", "PreheatOven",
    "SlideDishwasherRack", "SlideOvenRack", "SlideToasterOvenRack", "StartCoffeeMachine",
    "TurnOffMicrowave", "TurnOffSinkFaucet", "TurnOffStove", "TurnOnBlender",
    "TurnOnElectricKettle", "TurnOnMicrowave", "TurnOnSinkFaucet", "TurnOnStove",
    "TurnOnToaster", "TurnOnToasterOven", "TurnSinkSpout",
)

# The 235 ``pretrain/composite`` tasks: multi-stage, long-horizon kitchen activities.
# Episodes average ~1565 frames (78 s) against ~142 for atomic, so by frame count composite
# outweighs pretrain/atomic roughly 18:1 (27.6M vs 1.5M). Mixing the two unweighted therefore
# yields a ~95% composite corpus -- intentional here, but worth remembering when reading
# per-task numbers. Composite demos drive the mobile base far more than atomic ones, which is
# why they require ``use_base_action=True``.
DEFAULT_PRETRAIN_COMPOSITE_TASKS: tuple[str, ...] = (
    "AddIceCubes", "AddLemonToFish", "AddMarshmallow", "AddSugarCubes", "AddSweetener",
    "AdjustHeat", "AfterwashSorting", "AirDryFruit", "AlcoholServingPrep", "AlignSilverware",
    "ArrangeBreadBowl", "ArrangeBuffetDessert", "ArrangeDrinkware", "ArrangeTeaAccompaniments",
    "ArrangeUtensilsByType", "ArrangeVegetables", "AssembleCookingArray", "BalancedMealPrep",
    "BeverageOrganization", "BeverageSorting", "BlendIngredients", "BlendMarinade",
    "BowlAndCup", "BreadAndCheese", "BreadSetupSlicing", "BuildAppetizerPlate", "ButterOnPan",
    "CandleCleanup", "CerealAndBowl", "ChooseMeasuringCup", "ChooseRipeFruit", "CleanBoard",
    "CleanMicrowave", "ClearClutter", "ClearCuttingBoard", "ClearFoodWaste", "ClearFreezer",
    "ClearReceptaclesForCleaning", "ClearSink", "ClearSinkArea", "ClearSinkSpace",
    "CollectWashingSupplies", "ColorfulSalsa", "CondimentCollection", "CookieDoughPrep",
    "CoolBakedCake", "CoolKettle", "CreateChildFriendlyFridge", "CupcakeCleanup",
    "CutBuffetPizza", "DateNight", "DefrostByCategory", "DeliverBrewedCoffee", "DeliverStraw",
    "DessertAssembly", "DessertUpgrade", "DisplayMeatVariety", "DistributeChicken",
    "DivideBasins", "DivideBuffetTrays", "DrainVeggies", "DrinkwareConsolidation", "DryDishes",
    "DryDrinkware", "DumpLeftovers", "FillBlenderJug", "FillKettle", "FilterMicrowavableItem",
    "FoodCleanup", "FreezeBottledWaters", "FreezeCookedFood", "FreezeIceTray",
    "FreshProduceOrganization", "FryingPanAdjustment", "GarnishCupcake", "GatherCuttingTools",
    "GatherMarinadeIngredients", "GatherVegetables", "GetToastedBread", "HeatMug",
    "HeatMultipleWater", "HotDogSetup", "JuiceFruitReamer", "KettleBoiling",
    "LemonSeasoningFish", "LineUpCondiments", "LoadCondimentsInFridge", "LoadDishwasher",
    "LoadFridgeByType", "LoadFridgeFifo", "LoadPreparedFood", "MakeFruitBowl",
    "MakeLoadedPotato", "MatchCupAndDrink", "MaximizeFreezerSpace", "MealPrepStaging",
    "MeatSkewerAssembly", "MeatTransfer", "MicrowaveCorrectMeal", "MicrowaveDefrostMeat",
    "MicrowaveThawing", "MicrowaveThawingFridge", "MixCakeFrosting", "MixedFruitPlatter",
    "MoveFreezerToFridge", "MoveFridgeToFreezer", "MoveToCounter", "MoveToFreezerDrawer",
    "MultistepSteaming", "OrganizeBakingIngredients", "OrganizeCleaningSupplies",
    "OrganizeCoffeeCondiments", "OrganizeCondiments", "OrganizeMetallicUtensils",
    "OrganizeMugsByHandle", "OrganizeVegetables", "OvenBroilFish", "PackFoodByTemp",
    "PackFruitContainer", "PackIdenticalLunches", "PastryDisplay", "PlaceBeveragesTogether",
    "PlaceDishesBySink", "PlaceEqualIceCubes", "PlaceFoodInBowls", "PlaceIceInCup",
    "PlaceMeatInMarinade", "PlaceMicrowaveSafeItem", "PlaceStraw", "PlaceVegetablesEvenly",
    "PlaceVeggiesInDrawer", "PlateSteakMeal", "PlateStoreDinner", "PortionFruitBowl",
    "PortionInTupperware", "PortionOnSize", "PortionYogurt", "PreRinseStation", "PreSoakPan",
    "PreheatPot", "PrepForSanitizing", "PrepForTenderizing", "PrepFridgeForCleaning",
    "PrepMarinatingMeat", "PrepSinkForCleaning", "PrepareBroilingStation",
    "PrepareCheeseStation", "PrepareCocktailStation", "PrepareCoffee", "PrepareDishwasher",
    "PrepareDrinkStation", "PrepareSausageCheese", "PrepareSmoothie", "PrepareSoupServing",
    "PrepareStoringLeftovers", "PrepareToast", "PressChicken", "PrewashFoodAssembly",
    "PrewashFoodSorting", "QuickThaw", "RearrangeFridgeItems", "RecycleBottlesBySize",
    "RecycleSodaCans", "RecycleStackedYogurt", "RefillCondimentStation", "ReheatMeal",
    "RemoveBroiledFish", "RemoveCuttingBoardItems", "ReorganizeFrozenVegetables",
    "ResetCabinetDoors", "RestockBowls", "RestockCannedFood", "RestockPantry",
    "RestockSinkSupplies", "RetrieveIceTray", "RetrieveMeat", "ReturnHeatedFood",
    "ReturnWashingSupplies", "RinseBowls", "RinseCuttingBoard", "RinseSinkBasin", "RotatePan",
    "SanitizePrepCuttingBoard", "ScalePortioning", "ScrubBowl", "ScrubCuttingBoard",
    "SearingMeat", "SeasoningSpiceSetup", "SeasoningSteak", "ServeSteak", "ServeTea",
    "ServeWarmCroissant", "SetBowlsForSoup", "SetUpCuttingStation", "SetUpSpiceStation",
    "SetupBowls", "SetupButterPlate", "SetupFruitBowl", "SetupFrying", "SetupSodaBowl",
    "SetupWineGlasses", "ShakePan", "SimmeringSauce", "SizeSorting", "SnackSorting",
    "SoakSponge", "SortingCleanup", "SpicyMarinade", "StackBowlsCabinet", "StackBowlsInSink",
    "StackCans", "StartElectricKettle", "SteamInMicrowave", "StirVegetables",
    "StockingBreakfastFoods", "StoreDumplings", "StoreLeftoversByType", "StoreLeftoversInBowl",
    "StrainerSetup", "SweetSavoryToastSetup", "SweetenCoffee", "SweetenHotChocolate",
    "ThawInSink", "TiltPan", "ToastBagel", "ToastBaguette", "ToastOnCorrectRack",
    "TongBuffetSetup", "TransportCookware", "TurnOffSimmeredSauceHeat", "VeggieDipPrep",
    "WarmCroissant", "WashFish", "WashLettuce", "YogurtDelightPrep",
)

# Convenience union for the 300-task pretrain soup (atomic + composite). Both live under one
# converted root, so the loader still takes a single ``root``.
DEFAULT_PRETRAIN_ALL_TASKS: tuple[str, ...] = (
    DEFAULT_PRETRAIN_ATOMIC_TASKS + DEFAULT_PRETRAIN_COMPOSITE_TASKS
)

# Mobile-base action encoding (``use_base_action=True``). The Omron base is a
# JOINT_VELOCITY part with joints [forward, side, yaw, torso_height]; torso_height is
# never actuated in target/atomic (std 0 across all 18 tasks), so the base is a planar
# 3-DoF (x, y, yaw) system. Rather than regress the raw velocity command we derive a
# frame-wise EGO-FRAME base pose delta from ``observation.state`` (base_position +
# base_rotation) and encode it like the EEF delta (pos + rot6d). Because the deltas are
# expressed in the base frame at each step, no world->body yaw rotation is needed at
# inference; the closed-loop client converts back with
# ``base_motion[0:3] = (ego_delta / dt) / BASE_MAX_VELOCITY``.
#
# ``base_motion`` is a NORMALISED command in [-1, 1] (the JOINT_VELOCITY controller maps
# it onto the joint velocity limits), not a physical velocity. Calibrated on
# NavigateKitchen (least-squares through the origin of physical ego velocity against the
# recorded command, restricted to |command| > 0.2): full deflection corresponds to
# ~0.60 m/s forward, ~0.64 m/s lateral, ~1.25 rad/s yaw. Fit quality: yaw corr 0.997
# (6.9% residual); the translation channels are corr 0.92-0.96 with ~26% residual, which
# is controller lag — the commanded velocity is not reached instantaneously. Closed-loop
# feedback absorbs that per step.
BASE_MAX_VELOCITY: tuple[float, float, float] = (0.601, 0.640, 1.251)
_STATE_BASE_POS = slice(0, 3)  # absolute base position (world frame)
_STATE_BASE_ROT = slice(3, 7)  # absolute base rotation (quaternion, xyzw)
_CONTROL_MODE = slice(4, 5)  # +1 = base mode (arm tracks the moving base), -1 = arm mode

# ``base_encoding="raw"``: regress the native command instead of inverting the controller.
# The ego encoding above derives the base delta from the ACHIEVED pose in
# ``observation.state``, so recovering the command at inference requires inverting a
# controller with real inertia -- structurally lossy (yaw residual 6.9 %, translation
# 26 %), and a recorded demo can never be replayed exactly. Passing ``base_motion``
# through is an identity round-trip: the client writes ``env[7:11] = action[0:4]``.
#
# It also fixes a loss-weighting defect. Measured RMS per step: the ego translation delta
# is 0.018 (NavigateKitchen) to 0.002 (PickPlaceDrawerToCounter) in METRES, while the arm's
# ``eef_pos`` is a NORMALISED command with RMS 0.21-0.52 -- a 12x to 260x scale gap that,
# under ``action_normalization=None``, makes the base translation channels contribute
# ~1/144 to ~1/68000 of the arm's gradient. ``base_motion`` is normalised to [-1, 1]
# (RMS 0.49-0.61 on NavigateKitchen), i.e. the same scale as the arm block.
_BASE_MOTION = slice(0, 4)  # [forward, side, yaw, torso_height]; torso is never actuated


class RoboCasaLeRobotDataset(BaseActionLeRobotDataset):
    """RoboCasa manipulation dataset.

    Actions are ``[pos_delta(3), rot6d_delta(6), gripper(1)]`` (10D, arm only) or, with
    ``use_base_action=True``, widened to include the mobile base — 15D for
    ``base_encoding="raw"`` and 20D for ``"ego"``. Observation is a ``concat_view``
    composite selected by ``camera_set``. Reads v2.1 exports via the official lerobot
    backend; each of ``task_names`` is discovered under ``root`` as
    ``<root>/<task>/*/lerobot`` and registered as a separate shard.
    """

    # cosmos_robocasa: bundled normalization stats live next to the LIBERO file
    # (cosmos_framework/data/generator/action/normalizer_stats/); produced by
    # ``cosmos_framework/scripts/convert_robocasa_cosmos_policy_to_lerobot_v30.py --stats-out``.
    _ROBOCASA_NORMALIZERS_DIR: ClassVar[Path] = Path(__file__).parent.parent / "normalizer_stats"
    DEFAULT_ACTION_STATS_FILENAME: ClassVar[str] = "robocasa_cosmos_policy_frame_wise_relative_rot6d.json"

    def __init__(
        self,
        root: str,
        fps: float = 20.0,
        chunk_length: int = 16,
        split_seed: int = 42,
        split_val_ratio: float = 0.01,
        split: str = "train",
        mode: str = "wam",
        pose_convention: PoseConvention = "backward_framewise",
        rotation_format: str = "rot6d",
        action_normalization: ActionNormalization | None = None,
        tolerance_s: float = 1e-4,
        viewpoint: Viewpoint = "concat_view",
        task_names: tuple[str, ...] | list[str] = DEFAULT_ALL_ATOMIC_TASKS,
        use_state: bool = False,
        enable_fast_init: bool = False,
        camera_set: str = "wrist_lr",
        use_base_action: bool = False,
        base_encoding: str = "ego",
        episodes_per_task: int | None = None,
        episode_subset_seed: int = 42,
        use_image_augmentation: bool = False,
        jitter_after_compose: bool = False,
        action_stats_path: str | None = None,
        state_normalization: str | None = None,
        augment_crop_scale: float = 0.95,
        augment_rotation_deg: float = 0.0,
        augment_hue: float = 0.08,
    ) -> None:
        if rotation_format != "rot6d":
            raise NotImplementedError(f"RoboCasa loader only supports rotation_format='rot6d', got {rotation_format!r}.")
        # The CUDA torchcodec build needs the pip NVIDIA NPP libraries resolvable (cosmos_hs11: the LIBERO loaders get
        # this from ActionBaseDataset, this one derives from BaseActionLeRobotDataset which does not pre-load them).
        ensure_video_decoder_libs()
        # cosmos_robocasa: resolved by ``_normalizer_path`` (called from the base ctor when action_normalization is set).
        self._action_stats_path = action_stats_path
        super().__init__(
            fps=fps,
            chunk_length=chunk_length,
            split_seed=split_seed,
            split_val_ratio=split_val_ratio,
            split=split,
            mode=mode,
            embodiment_type="robocasa",
            viewpoint=viewpoint,
            pose_convention=pose_convention,
            rotation_format=rotation_format,
            action_normalization=action_normalization,
            tolerance_s=tolerance_s,
            enable_fast_init=enable_fast_init,
        )
        self._use_state = use_state
        # cosmos_hs11: train-time image augmentation, the DROIDLeRobotDataset recipe (= GR00T N1.5's RoboCasa365
        # data config: random crop to 95 % + resize back + ColorJitter 0.3/0.4/0.5/0.08). One parameter draw per
        # window is applied to every frame of every view (the views are stacked along T before the transform), so
        # the views keep agreeing on framing and lighting. Never applied to the val split: that loader must see the
        # same windows every pass. TOML: [dataloader_train].use_image_augmentation.
        if use_image_augmentation and split != "train":
            log.warning(f"RoboCasa: use_image_augmentation ignored for split={split!r} (train only).")
            use_image_augmentation = False
        self._use_image_augmentation = use_image_augmentation
        self._jitter_after_compose = jitter_after_compose
        self._image_augmentor: T.Compose | None = None
        self._color_augmentor: T.ColorJitter | None = None
        # cosmos_robocasa: augmentation geometry / strength. Defaults = the cosmos_hs11 numbers above (byte-identical
        # for the RoboCasa365 recipes). Cosmos Policy's RoboCasa "stronger" aug = random crop to 90 % of the AREA
        # (side 0.949 ~ 0.95) + a random rotation in [-5, 5] deg (black corners) + ColorJitter brightness 0.3 /
        # contrast 0.4 / saturation 0.5 / hue 0.05 -> ``augment_rotation_deg=5.0, augment_hue=0.05``.
        if not 0.0 < float(augment_crop_scale) <= 1.0:
            raise ValueError(f"augment_crop_scale must be in (0, 1], got {augment_crop_scale!r}.")
        if float(augment_rotation_deg) < 0.0:
            raise ValueError(f"augment_rotation_deg must be >= 0, got {augment_rotation_deg!r}.")
        self._augment_crop_scale = float(augment_crop_scale)
        self._augment_rotation_deg = float(augment_rotation_deg)
        self._augment_hue = float(augment_hue)
        # use_base_action: widen the arm-only 10D contract so mobile tasks
        # (NavigateKitchen et al.) are representable. ``base_encoding`` selects HOW:
        #
        #   "ego" (default, 20D) -- state-derived ego-frame base pose delta:
        #       [base_pos(3), base_rot6d(6), control_mode(1), eef_pos(3), eef_rot6d(6), gripper(1)]
        #   "raw" (15D)          -- the native base_motion velocity command, passed through:
        #       [base_motion(4), control_mode(1), eef_pos(3), eef_rot6d(6), gripper(1)]
        #   "raw_armfirst" (15D) -- the same channels as "raw", arm block FIRST (cosmos_hs11):
        #       [eef_pos(3), eef_rot6d(6), gripper(1), base_motion(4), control_mode(1)]
        #       dims 0..9 are then exactly the 10D ``[pos, rot6d, gripper]`` layout every Reptile meta-training
        #       embodiment (and LIBERO) uses, so a meta-initialized action-head row keeps its column semantics;
        #       the base command and the mode flag land on dims 10..14 (trained by YAM's right arm, mostly 0 / -1).
        #
        # "ego" is kept as the default so existing checkpoints/configs reproduce bit-for-bit.
        # See the ``BASE_MAX_VELOCITY`` block above for why "raw" is the better contract.
        self._use_base_action = use_base_action
        if base_encoding not in ("ego", "raw", "raw_armfirst"):
            raise ValueError(f"Unsupported base_encoding={base_encoding!r}. Use 'ego', 'raw' or 'raw_armfirst'.")
        self._base_encoding = base_encoding
        # Fixed data fraction (cosmos_hs11 RoboCasa365 recipes, docs/action_policy_robocasa_posttrain.md): keep only
        # ``episodes_per_task`` episodes of EACH task shard, picked by a seeded permutation of the shard's split
        # episodes in ``_filter_valid_episodes``. Prefixes of one permutation, so the 50-demo subset (the paper's 10 %)
        # is contained in the 150-demo subset (30 %); None = every split episode (100 %). The seeded 1 % val hold-out
        # of the base class is removed BEFORE this filter, so no subset ever overlaps the validation episodes.
        if episodes_per_task is not None and int(episodes_per_task) <= 0:
            raise ValueError(f"episodes_per_task must be a positive int or None, got {episodes_per_task!r}.")
        self._episodes_per_task = None if episodes_per_task is None else int(episodes_per_task)
        self._episode_subset_seed = int(episode_subset_seed)
        # cosmos_robocasa: the ``use_state`` token is an ABSOLUTE eef pose while the action normalizer's statistics
        # describe per-frame DELTAS, so once ``action_normalization`` is set the token needs its own statistics: the
        # ``"state"`` block of the same stats JSON (Cosmos Policy rescales its 9D proprio to [-1, 1] by min/max ->
        # ``state_normalization="minmax"``). ``_normalize_action_rows`` applies the two normalizers to row 0 / rows 1:.
        self._state_normalizer: ActionNormalizer | None = None
        if state_normalization is not None:
            if not use_state:
                raise ValueError("state_normalization requires use_state=True (there is no state token to normalize).")
            if use_base_action:
                raise ValueError("state_normalization supports the 10D arm-only contract only (use_base_action=False).")
            if state_normalization not in ("minmax", "quantile", "meanstd"):
                raise ValueError(f"Unsupported state_normalization={state_normalization!r}; use minmax/quantile/meanstd.")
            self._state_normalizer = resolve_action_normalization(
                state_normalization, self._load_state_norm_stats(), apply_forward_clamp=False
            )
        elif use_state and action_normalization is not None:
            raise ValueError(
                "use_state=True with action_normalization set needs state_normalization (otherwise the absolute-pose "
                "token would be scaled with delta-action statistics). Pass state_normalization='minmax'."
            )
        self._image_features = _IMAGE_FEATURES
        # camera_set:
        #   "wrist_lr"   (default): concat_view = wrist (top) + L/R (bottom, squished to half).
        #   "left_wrist"          : LIBERO-style = agentview_left + wrist, full-res, horizontally
        #                           concatenated ([T,C,H,2W]); right cam DROPPED, no downscaling.
        if camera_set not in ("wrist_lr", "left_wrist"):
            raise ValueError(f"Unsupported camera_set={camera_set!r}. Use 'wrist_lr' or 'left_wrist'.")
        self._camera_set = camera_set

        # Discover one LeRobot shard root per requested task under ``root``.
        self._all_shard_roots = self._discover_shard_roots(root, task_names)
        if not self._all_shard_roots:
            raise FileNotFoundError(
                f"No RoboCasa lerobot shards found under {root!r} for tasks {list(task_names)}. "
                f"Expected <root>/<task>/*/lerobot directories."
            )
        log.info(f"RoboCasaLeRobotDataset: {len(self._all_shard_roots)} task shard(s): {self._all_shard_roots}")

        # delta_timestamps: chunk_length per-frame action deltas; chunk_length+1
        # observation frames (one more image/state than transitions).
        observation_ts = [i * self._dt for i in range(0, self._chunk_length + 1)]
        action_ts = [i * self._dt for i in range(0, self._chunk_length)]
        self._delta_timestamps: dict[str, list[float]] = {_ACTION_FEATURE: action_ts}
        if self._use_state:
            self._delta_timestamps[_STATE_FEATURE] = observation_ts
        if self._camera_set == "left_wrist":
            # LIBERO-style: only agentview_left + wrist (right dropped).
            self._delta_timestamps[self._image_features["wrist"]] = observation_ts
            self._delta_timestamps[self._image_features["left"]] = observation_ts
        else:
            if self._viewpoint in ("wrist_view", "concat_view"):
                self._delta_timestamps[self._image_features["wrist"]] = observation_ts
            if self._viewpoint in ("third_person_view", "concat_view"):
                self._delta_timestamps[self._image_features["left"]] = observation_ts
                self._delta_timestamps[self._image_features["right"]] = observation_ts

        self._register_sources()

    @staticmethod
    def _discover_shard_roots(root: str, task_names: tuple[str, ...] | list[str]) -> list[str]:
        """Resolve ``<root>/<task>/*/lerobot`` for each requested task.

        ``root`` may point at the atomic dir (``.../target/atomic``) or directly
        at a single ``lerobot`` dir. Missing tasks are skipped with a warning.

        A task may hold more than one dated export (RoboCasa ships re-collected
        demos under a new date). All of them are registered: silently keeping one
        would train on less data than the caller staged, with nothing to show for it.
        """
        root = root.rstrip("/")
        if os.path.basename(root) == "lerobot" and os.path.isdir(os.path.join(root, "meta")):
            return [root]
        shard_roots: list[str] = []
        for task in task_names:
            matches = sorted(glob.glob(os.path.join(root, task, "*", "lerobot")))
            matches = [m for m in matches if os.path.isdir(os.path.join(m, "meta"))]
            if not matches:
                log.warning(f"RoboCasaLeRobotDataset: no lerobot shard for task {task!r} under {root!r}; skipping.")
                continue
            if len(matches) > 1:
                log.info(
                    f"RoboCasaLeRobotDataset: task {task!r} has {len(matches)} dated exports under "
                    f"{root!r}; registering all of them: {matches}"
                )
            shard_roots.extend(matches)
        return shard_roots

    def _filter_valid_episodes(self, meta: Any, episode_ids: list[int]) -> list[int]:
        """Base-class hook (``_append_index_records``), i.e. per shard = per task.

        ``episode_ids`` are the shard's split episodes (train = everything but the seeded 1 % val hold-out, i.e. ~497
        of the 502-513 target demos a task export holds). With ``episodes_per_task`` set, a permutation seeded by ``episode_subset_seed`` (the same
        seed for every shard) keeps its first N entries, so subsets of different sizes nest and never contain a val
        episode. Returns the ids sorted; the iterable shuffle decides the training order anyway.
        """
        n = self._episodes_per_task
        if n is None:
            return episode_ids
        where = getattr(meta, "root", "?")
        if n >= len(episode_ids):
            if n > len(episode_ids):
                log.warning(
                    f"RoboCasaLeRobotDataset: episodes_per_task={n} > {len(episode_ids)} split episodes in {where}; keeping all."
                )
            return episode_ids
        g = torch.Generator().manual_seed(self._episode_subset_seed)
        perm = torch.randperm(len(episode_ids), generator=g).tolist()
        kept = sorted(episode_ids[i] for i in perm[:n])
        log.info(
            f"RoboCasaLeRobotDataset: episodes_per_task={n} (seed {self._episode_subset_seed}): "
            f"kept {len(kept)} / {len(episode_ids)} split episodes of {where}"
        )
        return kept

    # ---- normalization (cosmos_robocasa) -----------------------------------

    def _normalizer_path(self) -> Path:
        """Stats JSON for ``action_normalization`` / ``state_normalization``.

        ``action_stats_path`` (absolute, or a bare filename under ``normalizer_stats/``) wins; the default is the
        bundled Cosmos Policy RoboCasa file. Blocks: ``global`` / ``global_raw`` (10D delta actions, the latter
        used by ``quantile_rot``) and ``state`` (10D proprioception token).
        """
        p = Path(self._action_stats_path) if self._action_stats_path else Path(self.DEFAULT_ACTION_STATS_FILENAME)
        if not p.is_absolute():
            p = self._ROBOCASA_NORMALIZERS_DIR / p
        return p

    def _load_state_norm_stats(self) -> dict[str, torch.Tensor]:
        raw = load_action_stats(str(self._normalizer_path()), stats_key="state")
        return {key: torch.from_numpy(value).float() for key, value in raw.items()}  # dict[str,[10]]

    def _normalize_action_rows(self, action: torch.Tensor) -> torch.Tensor:
        """Row 0 (the ``use_state`` token) with the state normalizer, rows 1: with the action normalizer."""
        state_normalizer = getattr(self, "_state_normalizer", None)
        if state_normalizer is None:
            return super()._normalize_action_rows(action)
        token, rows = action[:1], action[1:]  # [1,D], [chunk,D]
        return torch.cat([state_normalizer.normalize_action(token), super()._normalize_action_rows(rows)], dim=0)

    # ---- action / spec -----------------------------------------------------

    @property
    def action_dim(self) -> int:
        if self._use_base_action:
            if self._base_encoding in ("raw", "raw_armfirst"):
                # raw:          [base_motion(4), control_mode(1), eef_pos(3), eef_rot6d(6), gripper(1)]
                # raw_armfirst: [eef_pos(3), eef_rot6d(6), gripper(1), base_motion(4), control_mode(1)]
                return 15
            # [base_pos(3), base_rot6d(6), control_mode(1), eef_pos(3), eef_rot6d(6), gripper(1)]
            return 20
        return 10  # [pos(3), rot6d(6), gripper(1)]

    def _build_action_spec(self) -> ActionSpec:
        if self._use_base_action:
            if self._base_encoding == "raw_armfirst":
                # The 10D arm block first (= the LIBERO / meta-training layout), then the same base command
                # (``Pos(dim=4)``) and mode flag (``Gripper()``) as "raw" -- see the next branch for why.
                return build_action_spec(Pos(), Rot("rot6d"), Gripper(), Pos(dim=4), Gripper())
            if self._base_encoding == "raw":
                # base_motion is a NORMALISED velocity command in [-1, 1], the same units
                # as the arm's eef_pos block -- so it is declared as ``Pos`` (magnitude-based
                # idle detection, ``|v| < eps_t``) and joins the arm translation in one L2.
                # NOT ``Joint``: that branch is frame-DIFF based, so a constant cruise
                # command (diff == 0) would be misread as idle. The 4th channel
                # (torso height) is never actuated and contributes 0 to the norm.
                return build_action_spec(Pos(dim=4), Gripper(), Pos(), Rot("rot6d"), Gripper())
            # Base pose delta first (Pos+Rot), then the mode flag, then the arm block.
            # ``control_mode`` is a +/-1 channel regressed like the gripper (continuous
            # rectified flow, no discrete head); the client thresholds it at 0.
            return build_action_spec(Pos(), Rot("rot6d"), Gripper(), Pos(), Rot("rot6d"), Gripper())
        return build_action_spec(Pos(), Rot("rot6d"), Gripper())

    def _build_frame_wise_action(self, raw_action: torch.Tensor) -> torch.Tensor:
        """RoboCasa 12D per-frame action -> 10D ``[pos(3), rot6d(6), gripper(1)]``.

        Drops base_motion[0:4] + control_mode[4]; re-encodes the eef axis-angle
        delta to rot6d. Mirrors ``LIBEROLeRobotDataset._build_frame_wise_action``
        with RoboCasa slice indices.
        """
        raw = raw_action.float()  # [chunk, 12]
        translation = raw[:, _EEF_POS]  # [chunk, 3]
        rotation_matrix = convert_rotation(raw[:, _EEF_ROT], input_format="axisangle", output_format="matrix")
        rotation = convert_rotation(rotation_matrix, input_format="matrix", output_format="rot6d")  # [chunk, 6]
        gripper = raw[:, _GRIPPER]  # [chunk, 1]
        return torch.cat([translation, rotation, gripper], dim=-1)  # [chunk, 10]

    def _build_initial_state(self, state_seq: torch.Tensor) -> torch.Tensor:
        """Current EEF proprioception -> 10D ``[pos(3), rot6d(6), gripper(1)]`` token.

        Mirrors DROID's ``use_state`` path (``midtrain`` branch): the frame BEFORE the
        action chunk (index 0 of the ``chunk_length+1`` observation window) is prepended
        as a CLEAN conditioning action token — same width as the action so it flows
        through the shared ``action2llm`` embedding, gets ``sigma=0`` (never noised) and
        is excluded from the flow-matching loss (``condition_frame_indexes_action=[0]``).

        RoboCasa ``observation.state`` eef fields are in the (fixed) base frame; the
        rotation is a ``quat_xyzw`` re-encoded to rot6d to match the 10D action layout.
        Gripper = signed finger opening (``qpos[0]-qpos[1]``), a 1D summary of the
        two-finger state. NOTE: this token is ABSOLUTE pose while the predicted actions
        are per-frame deltas — identical to DROID ``midtrain`` — so pair with
        ``action_normalization=None`` (the quantile_rot delta-stats do not apply to an
        absolute pose).
        """
        s0 = state_seq[-self._chunk_length - 1].float()  # [16]; earliest = current pre-chunk state
        pos = s0[_STATE_EEF_POS]  # [3]
        quat = s0[_STATE_EEF_ROT].unsqueeze(0)  # [1,4] xyzw
        matrix = convert_rotation(quat, input_format="quat_xyzw", output_format="matrix")
        rot6d = convert_rotation(matrix, input_format="matrix", output_format="rot6d").reshape(6)  # [6]
        grip = s0[14:15] - s0[15:16]  # [1] signed finger opening
        return torch.cat([pos, rot6d, grip], dim=-1)  # [10]

    def _build_base_delta(self, state_seq: torch.Tensor) -> torch.Tensor:
        """Frame-wise EGO-FRAME base pose delta -> ``[chunk, 9]`` ``[pos(3), rot6d(6)]``.

        ``observation.state`` carries the base pose in the WORLD frame
        (``base_position``, ``base_rotation`` quat_xyzw). The delta between consecutive
        frames is re-expressed in the base frame at the earlier step, i.e.

            T_delta = T_base[t]^-1 @ T_base[t+1]

        so the representation is ego-centric and matches the ``backward_framewise``
        convention already used for the EEF. This removes any dependence on world
        heading: the closed-loop client can map straight back to the base velocity
        command with ``(ego_delta / dt) * BASE_VELOCITY_SCALE`` without a yaw rotation.

        The state window is ``chunk_length+1`` frames (one more than the transitions),
        so the ``chunk_length`` deltas line up one-to-one with the action chunk.
        """
        s = state_seq[-self._chunk_length - 1 :].float()  # [chunk+1, 16]
        pos = s[:, _STATE_BASE_POS]  # [chunk+1, 3] world
        quat = s[:, _STATE_BASE_ROT]  # [chunk+1, 4] xyzw, world
        rot = convert_rotation(quat, input_format="quat_xyzw", output_format="matrix")  # [chunk+1,3,3]

        r_prev = rot[:-1]  # [chunk,3,3]
        r_next = rot[1:]  # [chunk,3,3]
        # Translation delta rotated into the previous base frame.
        d_world = (pos[1:] - pos[:-1]).unsqueeze(-1)  # [chunk,3,1]
        d_ego = torch.matmul(r_prev.transpose(-1, -2), d_world).squeeze(-1)  # [chunk,3]
        # Relative rotation expressed in the previous base frame.
        r_rel = torch.matmul(r_prev.transpose(-1, -2), r_next)  # [chunk,3,3]
        rot6d = convert_rotation(r_rel, input_format="matrix", output_format="rot6d")  # [chunk,6]
        return torch.cat([d_ego, rot6d], dim=-1)  # [chunk, 9]

    def _assemble_action(self, arm: torch.Tensor, base_block: torch.Tensor, control_mode: torch.Tensor) -> torch.Tensor:
        """Order the three blocks per ``base_encoding``: arm first for ``raw_armfirst``, base first otherwise."""
        if self._base_encoding == "raw_armfirst":
            return torch.cat([arm, base_block, control_mode], dim=-1)  # [chunk, 15]
        return torch.cat([base_block, control_mode, arm], dim=-1)  # [chunk, 15 | 20]

    def _pad_state_token(self, token: torch.Tensor) -> torch.Tensor:
        """Widen the 10D EEF proprioception token to ``action_dim`` with zeros in the base block's position.

        The base block is zero-filled on purpose: under "ego" the absolute base pose is world-frame and the
        kitchen layout is re-randomised per episode, and under "raw" / "raw_armfirst" ``observation.state``
        carries no base VELOCITY at all -- only pose. Either way there is no transferable signal to put here,
        unlike the EEF pose, which is base-relative.
        """
        pad = torch.zeros(self.action_dim - 10, dtype=token.dtype)
        if self._base_encoding == "raw_armfirst":
            return torch.cat([token, pad], dim=-1)  # [action_dim]
        return torch.cat([pad, token], dim=-1)  # [action_dim]

    # ---- video -------------------------------------------------------------

    def _augment_views(self, views: list[torch.Tensor]) -> list[torch.Tensor]:
        """Random crop (95 %) + resize back (+ ColorJitter unless ``jitter_after_compose``) on every view at once.

        Mirrors ``DROIDLeRobotDataset._compose_multi_view``: the views are concatenated along T and pushed through one
        ``T.Compose`` call, so torchvision v2 samples ONE crop window / one set of colour factors per call and applies
        it to all frames of all views. Identity when augmentation is off, so existing runs are byte-identical.
        """
        if not self._use_image_augmentation:
            return views
        if self._image_augmentor is None:
            _, _, h, w = views[0].shape
            scale = getattr(self, "_augment_crop_scale", 0.95)
            transforms: list[Any] = [
                T.RandomCrop((max(1, int(h * scale)), max(1, int(w * scale)))),
                T.Resize((h, w), antialias=True),
            ]
            rotation_deg = getattr(self, "_augment_rotation_deg", 0.0)
            if rotation_deg > 0.0:
                # Cosmos Policy "stronger" aug: one angle in [-deg, deg] per window after the crop, size kept,
                # corners filled black (torchvision F.rotate expand=False, like theirs).
                transforms.append(T.RandomRotation(degrees=rotation_deg))
            color_jitter = T.ColorJitter(
                brightness=0.3, contrast=0.4, saturation=0.5, hue=getattr(self, "_augment_hue", 0.08)
            )
            if self._jitter_after_compose:
                self._color_augmentor = color_jitter
            else:
                transforms.append(color_jitter)
            self._image_augmentor = T.Compose(transforms)
        combined = self._image_augmentor(torch.cat(views, dim=0))
        out, start = [], 0
        for v in views:
            out.append(combined[start : start + v.shape[0]])
            start += v.shape[0]
        return out

    def _compose_multi_view(self, sample: dict[str, Any]) -> torch.Tensor:
        """Tile wrist (top) + left/right third-person (bottom) into one frame.

        Layout per frame (all source views are square, same size):
            ┌──────────────┐
            │    wrist      │   (H, W)
            ├───────┬──────┤
            │ left  │ right │   (H/2, W/2) each
            └───────┴──────┘
        Output height is 3H/2 (mirrors ``DROIDLeRobotDataset._compose_multi_view``).
        """
        wrist = sample[self._image_features["wrist"]]  # [T,C,H,W]
        left = sample[self._image_features["left"]]  # [T,C,H,W]
        right = sample[self._image_features["right"]]  # [T,C,H,W]
        wrist, left, right = self._augment_views([wrist, left, right])

        _, _, h_w, w_w = wrist.shape
        half_h, half_w = h_w // 2, w_w // 2
        left = F.interpolate(left, size=(half_h, half_w), mode="bilinear", align_corners=False)
        right = F.interpolate(right, size=(half_h, half_w), mode="bilinear", align_corners=False)
        bottom = torch.cat([left, right], dim=-1)  # [T,C,H/2,W]
        composite = torch.cat([wrist, bottom], dim=-2)  # [T,C,3H/2,W]
        if self._color_augmentor is not None:
            composite = self._color_augmentor(composite)
        return composite

    def _compose_left_wrist(self, sample: dict[str, Any]) -> torch.Tensor:
        """LIBERO-style: agentview_left (left) + wrist (right), full-res, horizontal.

        Both source views are native 256x256; concatenated along width -> [T,C,H,2W]
        (256x512). No downscaling, right camera dropped. Mirrors
        ``LIBEROLeRobotDataset`` concat_view ordering (third-person | wrist).
        """
        left = sample[self._image_features["left"]]  # [T,C,H,W]
        wrist = sample[self._image_features["wrist"]]  # [T,C,H,W]
        left, wrist = self._augment_views([left, wrist])  # crop per view (never across the tile seam)
        composite = torch.cat([left, wrist], dim=-1)  # [T,C,H,2W]
        if self._color_augmentor is not None:
            composite = self._color_augmentor(composite)
        return composite


    # ---- sample build ------------------------------------------------------

    def __getitem__(self, idx: int) -> dict[str, Any]:
        mode, _, _, sample = self._fetch_sample(idx)

        action = self._build_frame_wise_action(sample[_ACTION_FEATURE])  # [chunk, 10]

        if self._use_base_action:
            raw = sample[_ACTION_FEATURE].float()
            control_mode = raw[:, _CONTROL_MODE]  # [chunk, 1]; +/-1, regressed like the gripper
            if self._base_encoding in ("raw", "raw_armfirst"):
                # Native normalised velocity command, passed through unchanged -> [chunk, 15].
                base_block = raw[:, _BASE_MOTION]  # [chunk, 4]
            else:
                # State-derived ego-frame pose delta -> [chunk, 20].
                base_block = self._build_base_delta(sample[_STATE_FEATURE])  # [chunk, 9]
            action = self._assemble_action(action, base_block, control_mode)

        # EEF proprioception: prepend the current eef pose as a clean conditioning
        # token (DROID use_state parity). Compute idle_frames on the delta-only chunk
        # BEFORE prepending so the absolute-pose token does not skew idle detection.
        state_extras: dict[str, Any] = {}
        if self._use_state:
            idle = self._compute_idle_frames(action)
            if idle is not None:
                state_extras["idle_frames"] = idle
            initial_state = self._build_initial_state(sample[_STATE_FEATURE])  # [10]
            if self._use_base_action:
                # Widen the conditioning token to the action contract (20D ego / 15D raw / raw_armfirst);
                # the base block is zero-filled, see ``_pad_state_token``.
                initial_state = self._pad_state_token(initial_state)  # [action_dim]
            action = torch.cat([initial_state.unsqueeze(0), action], dim=0)  # [chunk+1, action_dim]

        if self._skip_video_loading:
            video = None
        elif self._camera_set == "left_wrist":
            video = self._compose_left_wrist(sample)
        elif self._viewpoint == "concat_view":
            video = self._compose_multi_view(sample)
        elif self._viewpoint == "wrist_view":
            video = sample[self._image_features["wrist"]]
        else:  # third_person_view
            video = sample[self._image_features["left"]]

        ai_caption = sample["task"]

        extras: dict[str, Any] = {}
        if self._camera_set == "left_wrist":
            extras["additional_view_description"] = (
                "The left half is a third-person view of the scene. "
                "The right half is from the wrist-mounted camera."
            )
        elif self._viewpoint == "concat_view":
            extras["additional_view_description"] = (
                "The top row is from the wrist-mounted camera. "
                "The bottom row contains two horizontally concatenated third-person views of the scene."
            )
        return self._build_result(
            mode=mode, video=video, action=action, ai_caption=ai_caption, **state_extras, **extras
        )
