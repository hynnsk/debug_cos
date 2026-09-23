# hs12: masked V-JEPA feature prediction for 30-demo LIBERO-10

`cosmos_hs10`의 현재 소스와 `launch_sft_action_policy_libero_10_edge_repa.sh`를 기반으로 만든 실험이다.
목표는 mid-trained Cosmos3-Edge의 action post-training에 **가려진 영상의 feature 예측**을 보조 과제로 추가하는 것이다.
학습 성능 개선은 가설이며, 실제 LIBERO rollout 성공률로 검증해야 한다.

## 설계와 원래 아이디어의 보완

V-JEPA 2.1의 공식 학습은 masked target뿐 아니라 context target도 예측하고, 여러 encoder depth에
supervision을 적용한다. 아래 구현은 이 원리를 가져온 **frozen-teacher masked distillation**이다.
Cosmos의 EMA를 V-JEPA teacher로 사용하는 방식이나 V-JEPA 2.1 전체 학습법의 재현은 아니다.

- [공식 설명 및 논문 링크](https://github.com/facebookresearch/vjepa2#v-jepa-21-pre-training)
- [공식 target normalization / masked + context regression 코드](https://github.com/facebookresearch/vjepa2/blob/main/app/vjepa_2_1/train.py#L550-L655)

가장 중요한 차이는 마스크를 적용하는 위치다. V-JEPA는 patch embedding 후 visible token을 선택하지만,
Cosmos의 입력은 이미 VAE가 주변 시공간을 섞은 latent이다. 원본 전체 영상을 VAE에 넣은 다음 latent만
지우면 다른 latent에 숨긴 픽셀의 정보가 남는다. 따라서 **native RGB를 먼저 가리고, resize와 reflection
padding을 다시 수행한 뒤 frozen VAE로 재인코딩**한다. 가려진 latent query 위치는 MoT 안에 유지된다.
MoT 앞 8개 block이 context를 이용해 이 위치의 representation을 만들고 작은 MLP가 teacher 차원으로 투영한다.

동일 공간의 과거/미래 프레임을 그대로 읽는 쉬운 경로를 줄이기 위해 마스크는 clip 전체에 같은 위치로 유지하는
spatial tube다. 첫 conditioning frame 0도 가린다. 두 카메라는 독립적으로 마스크를 샘플링하며,
한 카메라에서 다른 카메라를 통해 유추하는 것은 허용한다. 원본 영상의 masked 영역을 직접 읽는 것과
주변 context로 해당 영역을 추론하는 것은 다르며, 후자는 이 과제의 목적이다.

보조 branch에는 action, proprioception, task caption을 넣지 않는다. packing에 필요한 BOS/EOS/시작 기호만
남긴다. 원래 branch의 정답 action이 섞인 noised action이나 깨끗한 영상 latent를 재사용하지 않는다.
두 branch는 독립적으로 pack되며 서로 attention하지 않는다. 보조 branch에는 FM loss를 적용하지 않는다.

## 데이터 흐름과 loss

```text
기존 branch: original video + noised action -> MoT 전체 -> 기존 video/action FM loss

보조 student: native RGB -> tube mask -> resize/pad -> frozen VAE
              -> MoT block 1..8 -> small MLP -> p_i
보조 teacher: unmasked native future RGB -> frozen V-JEPA 2.1 EMA encoder
              -> fixed per-view avgpool -> channel LayerNorm -> y_i [stop-gradient]

L_total = L_original_FM + lambda(step) * L_JEPA
L_JEPA  = mean_sample( mean_masked(|p-y|) + alpha * mean_context(|p-y|) )
lambda(step) = loss_weight * min(1, step / masked_warmup_steps)
```

feature 차원 평균도 L1 mean에 포함된다. teacher만 channel LayerNorm하고 student 출력은 그대로 회귀한다.
target은 avgpool과 LayerNorm을 포함하여 완전히 detach된다. 학습 가능한 target adapter는 허용하지 않는다.
mask/context loss를 각각 평균하여 mask 비율에 따라 loss 크기가 불필요하게 바뀌는 것을 줄인다.
rank별 auxiliary sample 수가 다르면 gradient averaging에 맞춰 global sample mean으로 보정한다.

teacher는 기존 ViT-B/16 checkpoint의 `ema_encoder`를 고정해서 사용한다. 매 학습 step에 teacher를
Cosmos로부터 EMA 업데이트하지 않는다. `[model.ema]`는 기존 Cosmos weight EMA이며 이 teacher와 별개다.
teacher는 17개 프레임 중 미래 16개를 사용한다. student에는 마스킹한 frame 0을 포함한 17개 프레임이 들어간다.

정렬은 hs10과 같이 view별 teacher `8×16×16`을 `4×5×5`로 평균내고,
두 view를 width 방향으로 합쳐 `4×5×10`으로 만든다. sample → time → height → width 순서가 일치한다.
conditioning latent frame 0에는 직접 teacher loss가 없지만 context로 gradient가 전달될 수 있다.
VAE receptive field와 teacher tubelet이 동일하지 않으므로 이 정렬은 공간/시간에 대한 근사이며,
정확히 같은 receptive field의 patch를 맞추는 것은 아니다.

`16→5` adaptive pooling은 경계에서 bin이 겹친다. 선택된 masked target bin에 포함되는 **모든 teacher
patch의 픽셀**을 가려 경계에 원본 픽셀이 남지 않도록 한다. 그 결과 실제 가린 픽셀 비율은 설정한 target-cell
비율보다 높을 수 있고, context target cell 일부도 경계에서 가려질 수 있다. `jepa_mask_fraction`은 픽셀 비율이
아닌 target-cell 비율이다. 픽셀 context가 완전히 사라지는 마스크는 다시 샘플링한다.

기본 마스크 범위는 target cell의 40–70%다. 이는 작은 `5×5` grid와 30-demo 상황에 맞춘 출발점이며
공식 V-JEPA mask distribution을 복제한 값이 아니다. full future context에서 가린 공간을 복원하는 과제이므로,
현재 관측만으로 미래를 예측하는 causal forecasting objective와도 구분해야 한다.

## 왜 기존 V-JEPA predictor를 가져오지 않았나

MoT 앞 8개 block이 이미 token 사이의 정보를 섞는다. 따라서 각 masked 위치의 contextual feature를
작은 MLP로 투영하는 것만으로 유효한 prediction objective를 만들 수 있다. 큰 predictor를 추가하면 30개의
demonstration에서 predictor가 대부분의 적응을 담당하고 MoT feature의 개선이 작아질 가능성이 있다.
또한 pretrained predictor는 encoder feature 분포, positional encoding, grid, depth별 target 규약과 짝이
맞아야 한다. MoT feature를 차원만 맞춰 넣는 것이 곧 pretrained predictor의 올바른 재사용은 아니다.

기본은 hidden width 512의 기존 3-layer SiLU MLP이다. predictor 용량이 표현 학습을 대신하는지 보려면
`linear` variant와 비교한다. 추가 pretrained predictor dependency나 checkpoint는 필요하지 않다.

## 실행

기존 Cosmos 환경에서 실행한다. 새 디렉터리에는 hs10의 출력 checkpoint나 가상환경을 복사하지 않았으므로
실제 데이터/VAE/base checkpoint의 **절대 경로**를 설정한다. 현재 디렉터리의 패키지가 import되도록 한다.

```bash
cd /gallery_moma/hyunseok.seong/project/cosmos_hs12/packages/cosmos3
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH=''
export LIBERO_ROOT=/absolute/path/to/libero_10
export BASE_CHECKPOINT_PATH=/absolute/path/to/Cosmos3-Edge
export WAN_VAE_PATH=/absolute/path/to/Wan2.2_VAE.pth
export COSMOS_STORAGE=/absolute/path/to/cosmos_storage

NPROC_PER_NODE=2 bash examples/launch_sft_action_policy_libero_10_edge_masked_jepa.sh
```

launcher는 기존 REPA launcher를 재사용하며 `MASTER_PORT=50018`, `NPROC_PER_NODE=2`를 기본으로 쓴다.
`REPA_TOML_FILE`, `EXTRA_TAIL_OVERRIDES`, checkpoint/environment override는 기존과 같다.
기존 `launch_sft_action_policy_libero_10_edge_repa.sh` 자체의 default는 hs10 REPA 비교군으로 유지했다.

| `JEPA_VARIANT` | 학습 목표 | 용도 |
|---|---|---|
| `dense` (기본) | masked L1 + 0.25 × context L1, MLP | 권장 시작점 |
| `masked_only` | masked L1, MLP | context supervision 기여도 |
| `linear` | masked L1 + 0.25 × context L1, linear | predictor 용량 기여도 |

```bash
JEPA_VARIANT=masked_only bash examples/launch_sft_action_policy_libero_10_edge_masked_jepa.sh
JEPA_VARIANT=linear bash examples/launch_sft_action_policy_libero_10_edge_masked_jepa.sh
```

기본 recipe는 2000 steps, primary batch 최대 128 windows/rank, auxiliary 최대 16 windows/rank,
block 8, `loss_weight=0.5`, warmup 200 steps이다. 현재 Edge의 video/action FM scale은 각각 10이므로,
0.5는 그 scale 위에 더하는 값이다. 기존 cosine loss의 weight 5와 수치만으로 비교하면 안 된다.
auxiliary branch는 8번째 block에서 끝나므로 그 이후 decoder와 action head를 추가 실행하지 않는다.
시간/메모리 사용량은 실제 GPU에서 측정해야 한다.

먼저 짧은 GPU smoke를 실행할 때에는 전체 실험과 다른 output root를 사용한다.
warmup을 0으로 둬 2 step 안에도 auxiliary gradient가 실제로 적용되게 한다.

```bash
OUTPUT_ROOT="$PWD/outputs/hs12_smoke" \
EXTRA_TAIL_OVERRIDES='trainer.max_iter=2 trainer.run_validation=false trainer.run_validation_on_start=false checkpoint.save_iter=2 job.wandb_mode=disabled model.config.repa.masked_warmup_steps=0 model.config.repa.masked_max_samples=2 dataloader_train.max_samples_per_batch=2' \
bash examples/launch_sft_action_policy_libero_10_edge_masked_jepa.sh
```

그 다음 동일한 smoke output에서 `trainer.max_iter=3`으로 재실행하면 기존 DCP auto-resume 경로를 점검할 수 있다.
projector는 `net.repa_head.*` 아래에 있으므로 기존 optimizer allowlist, EMA, checkpoint 경로에 포함된다.
mid-trained checkpoint warm start에서만 `repa_`를 skip하고, 실제 resume에서는 저장된 projector를 복원한다.
variant별 projector shape가 다를 수 있으므로 서로 다른 variant는 별도 job/output을 사용한다.

## 30-demo 조건과 평가

모든 variant의 train subset은 `libero_10_3ep_per_task_seed42.json`이다. 10 tasks × 3 episodes = 30 demos이며,
validation은 train과 겹치지 않는 50개 episode이다. auxiliary video도 **현재 train batch에서만** 가져오며
남은 demonstration을 추가 visual pretraining 데이터로 사용하지 않는다. validation은 no-grad로만 실행된다.

기존 recipe의 action normalizer는 bundled `quantile_rot` 통계를 그대로 사용한다. 이 통계를 30-demo subset에서
새로 계산하지 않았으므로, "통계 추정까지 오직 30 demos만 사용"하는 엄격한 조건을 자동으로 만족한다고
주장하면 안 된다. 엄격한 실험에서는 train 30 demos로 계산한 별도 stats를 train/validation dataset의
`action_stats_path`와 policy server의 `--action-stats-path`에 동일하게 지정해야 한다.
기존 코드와 비교할 때에도 normalizer 조건은 모두 같게 유지한다.

훈련용 seed와 고정 episode subset의 seed는 별개다. `trainer.seed`만 바꾸어도 같은 30 demos를 사용한다.
다른 demonstration 선택에 대한 민감도까지 보려면 기존 `make_libero_episode_subset.py`로 train/held-out
subset을 함께 다시 만들고, 다른 실험 결과와 섞이지 않게 저장한다. 마스크 RNG는 step/rank/`masked_seed`로
결정하며 main FM RNG를 소비하지 않는다. 같은 step의 gradient-accumulation microbatch는 같은 mask schedule을
사용한다. validation에서는 checkpoint iteration과 무관한 mask schedule을 사용한다.

최소 비교는 같은 subset, optimizer, step 수, primary batch의 (1) FM only, (2) 기존 REPA,
(3) masked JEPA dense, (4) masked-only이다. linear는 표현 개선이 projector에만 집중되는지 보는 추가 비교다.
FM-only는 기존 REPA recipe에 `model.config.repa.enabled=false` override를 적용하면 된다.
보조 경로에 추가 compute가 있으므로 동일 step 비교와 wall-clock 비교를 구분해서 보고한다.
checkpoint 선택은 held-out 지표로 하고 최종 test rollout 성공률을 별도로 평가한다.

| 로그 | 해석 |
|---|---|
| `jepa_masked_loss` | 가린 target의 feature L1 |
| `jepa_visible_loss` | context target의 feature L1; masked-only에서도 모니터링 |
| `jepa_loss` | masked + alpha × context, weight 적용 전 |
| `jepa_weighted_loss`, `jepa_weight` | 실제 가중 기여도와 warmup weight |
| `jepa_centered_cos` | sample별 공간/시간 평균을 제거한 masked-token cosine |
| `jepa_pred_std` | sample 안 token 간 feature 표준편차; 상수 예측 감지 |
| `jepa_mask_fraction` | masked target-cell 비율 |

train/`val/`의 masked loss가 내려가도 action 개선을 보장하지 않는다. centered cosine과 예측 분산이 거의 0이면
정적인 평균 feature를 내는 해법을 의심할 수 있다. 그러나 이 진단만으로 task/position별 평균을 외운 모든
shortcut을 검출할 수 있는 것은 아니다. action FM validation과 LIBERO rollout 성공률을 함께 확인한다.

## 검증 범위

CPU 테스트는 숨긴 픽셀을 바꿔도 masked canvas와 mixing encoder 입력이 동일한지, pooling 경계를 완전히 가리는지,
두 view의 순서, teacher detach, masked/context loss와 gradient, 실제 packer, 30/50 demo split 비중복을 검증한다.
또 실제 `Cosmos3VFMNetwork.forward`와 MoT eager loop를 작은 CPU decoder에 연결해 auxiliary 추가 전후의
primary action/video 출력 일치, 앞 8개 block만 받는 auxiliary gradient, activation checkpoint 재계산,
exception 후 capture 상태 복구를 검증한다. TOML validation과 training entrypoint의 dryrun도 포함한다.

```bash
LD_LIBRARY_PATH='' OMP_NUM_THREADS=1 python -m pytest --confcutdir=cosmos_framework -o addopts='' \
  cosmos_framework/model/generator/repa/masked_prediction_test.py \
  cosmos_framework/model/generator/repa/masked_network_test.py \
  cosmos_framework/model/generator/repa/alignment_test.py \
  cosmos_framework/model/generator/repa/adapters_test.py \
  cosmos_framework/model/generator/repa/vjepa_teacher_test.py \
  cosmos_framework/configs/toml_config/repa_toml_test.py -q
```

작성 환경에 CUDA GPU가 없어서 실제 VAE/teacher/base checkpoint를 함께 사용하는 FSDP 학습, GPU peak memory,
DCP 저장/재시작, LIBERO rollout 성능은 실행 검증하지 못했다. 현재 지원 범위는 기존 LIBERO concat-view,
single-clip, conditioning frame `[0]`, context parallel degree 1이다. multi-item/camera-major layout은 거부한다.
새 branch가 inference에 호출되지 않으므로 학습 후 policy 실행에 teacher나 mask가 필요하지 않다.

## 주요 코드

- `model/generator/repa/masked_prediction.py`: 마스크, pixel canvas, loss, auxiliary batch
- `model/generator/mot/cosmos3_vfm_network.py`: 한 root forward 안의 auxiliary/main branch
- `model/generator/mot/unified_mot.py`: block-k early exit (checkpointed block 밖에서 처리)
- `model/generator/omni_mot_model.py`: teacher 및 batch 준비, warmup, total loss와 metric 연결
- `examples/toml/sft_config/action_policy_libero_10_edge_masked_jepa*.toml`: 세 실험 recipe

위 `model/...` 경로는 `cosmos_framework/` 아래이다.
