2026-10-02 기준 `cosmos_hs11` 연구·구현 검토입니다. 작업 공간의 원본 코드는 `cosmos/packages/cosmos3`에 있었으며, 요청한 두 TOML, 실제 저장된 실행 설정, 데이터 parquet, Reptile 학습 로그와 기존 rollout 결과를 함께 확인했습니다. 학습 코드는 수정하지 않았고 GPU를 사용하지 않았습니다.

**판단: 초기화 개선 방향은 유효한 실험 신호가 있습니다. 그러나 그 개선을 Reptile의 few-shot 적응 능력으로 설명하는 근거, 그리고 DINO 정렬이 action 일반화를 개선한다는 근거는 아직 부족합니다. 현재 가장 필요한 것은 새 손실을 늘리는 것보다 진단·대조군·평가 절차를 정비하는 일입니다.**

아래에서 ‘확인’은 현재 코드/저장된 파일로 확인한 내용이고, ‘가설’은 추가 실험이 필요한 해석입니다. 현행 코드가 모든 과거 실험에 그대로 사용되었다고 보장할 수는 없으므로, 가능하면 저장된 `config.yaml`과 로그를 우선했습니다. 수치와 원본 파일 경로는 [audit_summary.json](audit_summary.json), 재집계 코드는 [collect_evidence.py](collect_evidence.py)에 있습니다.

1. **실제 실행 내용과 설명 사이에 차이가 있습니다.**

| 항목 | 확인한 실제 값 |
|---|---|
| LIBERO-10 원본 | 379 episodes, 101,469 frames, 10 tasks |
| 학습 subset | seed 42, 정확히 task당 3개, 총 30개; parquet task ID와 대조 및 난수 선택 재현 완료 |
| 학습 window | 길이 16의 action + 17 video frames; 유효 window 7,618개 |
| 검증 subset | task당 5개, 총 50개; 학습 subset과 교집합 없음; 유효 window 13,049개 |
| Reptile source | fractal, bridge, robomind_ur, robomind_franka, molmoact2_yam |
| Reptile support/query | source 하나에서 support 16 demos × 16 windows = 256; query 8 × 16 = 128 |
| Reptile inner | GPU 2개 기준 rank당 128 windows, 10 Adam steps, LR 5e-5, warmup 2, episode마다 Adam reset |
| Reptile outer | 1,000 iterations, source 하나씩, SGD interpolation ε 약 0.5→0.05 |
| LIBERO post-training | rank당 128 × 2 = global batch 256, 2,000 steps, warmup 200, LR 5e-5 |
| DINO auxiliary | ViT-B/14, future 16 frames, 두 카메라별 독립 encoding, MLP projection, avgpool, weight 5 |
| 정렬 위치 | **28개 중 24번째 블록 출력, 최종 norm 이전**; 마지막 28번째 블록이 아님 |

`edge2`의 이름은 `k8`인데 실제 K는 16이고, TOML에는 아직 4-GPU/20-step/8th-block/4096-val-window 등의 오래된 주석이 남아 있습니다. v2의 실제 검증량은 `16 batches × 16 windows × 2 ranks = 512 windows`입니다. 이런 차이는 연구 기록과 대조군 설계를 틀리게 만들므로 주석·run name부터 정리하는 것이 좋습니다. [edge2 TOML:64][meta-toml], [v2 TOML:42][post-toml], [backbone JSON:6][backbone].

‘full-parameter’도 모델 전체 약 3.37B를 모두 학습한다는 의미는 아닙니다. 이 코드에서는 Cosmos의 full-FT recipe에 해당하는 generation pathway와 projection/action heads 등 약 1.42B를 학습하고 understanding pathway 등은 동결합니다. 이 선택은 합리적이며 명칭을 정확히 설명하면 됩니다. [trainable keys:38][theta-keys].

2. **저장된 결과에는 개선 신호가 있지만, 구성 요소의 효과를 분리해서 해석해야 합니다.**

모두 아래 파일 이름의 iteration 2000, task당 50회, 총 500회 rollout 결과입니다. `50demo`는 여기서는 평가 trial 수를 뜻하며, 학습 demonstration 수가 아닙니다.

| 저장된 실험 | 성공 수 | SR |
|---|---:|---:|
| `hs08_v2_50demo_2000` | 290/500 | 58.0% |
| `hs10_repa_dinov2b_l24_avgpool_w5.0_50demo_2000` | 297/500 | 59.4% |
| `edge_libero10_3ep_fullft_reptileinit_50demo_2000` | 322/500 | 64.4% |
| `edge_libero10_3ep_fullft_reptileinit_repa_dinov2b_l24_avgpool_w5.0_50demo_2000` | 331/500 | 66.2% |

초기화 변경의 관측 차이는 +6.4%p, Reptile 이후 DINO 추가 차이는 +1.8%p입니다. 다만 앞의 두 실험은 다른 repository의 결과이고 기본 head LR multiplier도 5× 대 1×입니다. 이를 엄밀하게 통제된 2×2 ablation이라고 부를 수는 없습니다. 단순 산술로 계산한 interaction도 `(66.2−64.4)−(59.4−58.0)=0.4%p`로 작아서 ‘Reptile과 DINO의 특별한 시너지’는 아직 입증되지 않았습니다. 각 원본 summary 경로와 저장 config 비교는 [audit_summary.json](audit_summary.json)에 있습니다.

Reptile-only 대비 v2의 task별 변화는 다음과 같습니다. **LeRobot dataset task ID와 simulator task ID의 순서가 다르므로 이름으로 대응시켰습니다.**

| simulator task | 설명 | Reptile | + DINO v2 | 차이 |
|---:|---|---:|---:|---:|
| 0 | soup + tomato sauce → basket | 28% | 26% | −2%p |
| 1 | cream cheese + butter → basket | 64% | 74% | +10%p |
| 2 | stove on + moka pot | 42% | 60% | +18%p |
| 3 | black bowl → drawer, close | 64% | 74% | +10%p |
| 4 | two mugs → two plates | 56% | 66% | +10%p |
| 5 | book → caddy | 84% | 66% | −18%p |
| 6 | mug + chocolate pudding | 68% | 70% | +2%p |
| 7 | soup + cream cheese → basket | 76% | 64% | −12%p |
| 8 | both moka pots → stove | 72% | 80% | +8%p |
| 9 | mug → microwave, close | 90% | 82% | −8%p |

episode 번호로 짝지으면 실패→성공 70건, 성공→실패 61건입니다. **과거 평가의 초기 상태와 설정이 실제로 동일했다는 조건하에** exact McNemar p≈0.485입니다. 따라서 이 한 쌍으로 DINO의 이득이 확실하다고 결론내리기 어렵습니다. 이는 DINO가 효과 없다는 증명도 아닙니다. summary에는 checkpoint 경로, seed, initial-state hash, server sampling 설정이 없어 동일 조건의 역사적 실행인지 완전히 확인할 수 없습니다. [summary 생성부:1317][eval-summary].

3. **우선 확인할 핵심 신호: Reptile이 학습 후반에 adaptation으로 query 성능을 악화시킵니다.**

`edge2`의 실제 1,000 iteration 로그를 네 구간으로 나눴습니다. query는 5 iteration마다 평가되므로 각 구간 50회입니다. 값은 action-only가 아닌 현재 설정의 weighted total loss입니다.

| meta iteration | adaptation 전 query 평균 | 10-step 후 평균 | 악화 횟수 |
|---|---:|---:|---:|
| 1–250 | 2.6665 | 2.4531 | 27/50 |
| 251–500 | 1.9679 | 2.1832 | 41/50 |
| 501–750 | 2.0794 | 2.3428 | 42/50 |
| 751–1000 | 1.8125 | 2.0644 | 43/50 |

전체적으로 initialization 자체의 source loss는 낮아지고 있지만, 거기서 10-step adaptation한 모델의 query loss는 오히려 나빠지는 경향입니다. 현재 실험은 ‘좋은 robot pretraining’이라는 설명과도 양립합니다. ‘빠르게 적응하기 좋은 초기화’라는 설명을 지지하려면 adaptation curve를 따로 검증해야 합니다. 원본: [reptile_train_log.jsonl:1][meta-log].

다만 이 로그에도 측정상의 한계가 있습니다. adaptation 전/후 `_eval_query()`는 같은 clean query window를 쓰지만 호출마다 sigma/noise를 다시 샘플합니다. 따라서 각각의 gain에는 Monte Carlo 오차가 섞이고, query를 평가하는 행위 자체가 이후 학습의 global RNG도 소비합니다. source query transform에는 CFG dropout 0.1도 적용됩니다. [query 평가:373][meta-query], [noise 재샘플:1497][noise], [source transform:178][source-transform].

권장 수정은 query 전용 RNG를 분리하고, 같은 `(window, sigma, noise)`로 adaptation 전/후를 비교하는 것입니다. K=3 또는 task-balanced 3-shot support에 대해 0/1/5/10/20-step action loss와 video loss를 분리해서 기록하십시오. source task/embodiment를 고정 holdout한 meta-validation도 필요합니다. 현재 query는 episode 내부에서 support와 disjoint이지만, 다음 meta episode에서 support로 다시 등장할 수 있으므로 독립적인 meta-validation set은 아닙니다.

4. **Reptile 수식은 맞지만, 현재 episode 설계는 downstream 3-shot protocol과 상당히 다릅니다.**

실제 outer update는 `theta += eps * (adapted_theta − theta)`이며, fp32 local shard의 사본을 유지하고 model에 복원합니다. optimizer state와 FusedAdam group step도 초기화합니다. 이 핵심 동작을 잘못 구현했다고 볼 근거는 찾지 못했습니다. [interpolation:231][interpolation], [optimizer reset:103][reset].

문제는 무엇을 하나의 meta-task로 정의하느냐입니다. 현재 sampler는 embodiment 하나를 고른 후 **그 데이터셋 전체에서 demonstration 16개를 균등 추출**합니다. 같은 language task에서 3개씩 선택하는 구조가 아니고 task balance도 없습니다. downstream은 10개 task의 3개 demonstration을 합쳐 한 정책을 학습합니다. ‘새 로봇 domain의 혼합 작업에 적응’이라는 현재 source objective와 ‘각 target task에 demonstration 3개씩 있는 multi-task adaptation’의 차이를 설명하거나 줄여야 합니다. [sampler:205][sampler].

또한 `16 × 16 / 2 = 128`이라 rank당 support batch가 딱 하나이고, 그 **동일한 256개 window를 10번 반복**합니다. sigma/noise는 새로 뽑으므로 완전히 같은 noisy input을 재학습하는 버그는 아닙니다. Reptile 역시 full batch 사용으로 자동 무효가 되는 것은 아닙니다. 다만 서로 다른 demonstration/window에서 얻은 gradient가 서로 도움이 되도록 유도하는 효과를 확인하려면, 16 demonstrations는 유지하면서 inner step마다 작은 task-balanced minibatch를 다시 뽑는 대조가 필요합니다. 단순히 per-rank batch 크기만 낮추면 현재 코드는 demonstration별로 이어 붙인 순서를 순환하므로, batch를 만드는 단계에서 demo/task를 섞어야 합니다. [inner loop:424][inner-loop], [support packing:550][packing].

원 Reptile 논문은 inner batch 구성과 gradient 간 관계를 분석하며, 실험에서는 Adam β1=0을 사용했습니다. 현재 β1=0.9와 episode별 state reset은 downstream을 흉내 내는 하나의 설계 선택이지, 그 자체로 오류는 아닙니다. 이 설정에서 별도의 검증이 필요합니다. 참고: [On First-Order Meta-Learning Algorithms](https://arxiv.org/html/1803.02999v3).

5. **가장 중요한 대조군은 같은 source data를 이용한 일반 SFT입니다.**

Mid-trained Edge → LIBERO와 Reptile → LIBERO의 차이에는 추가 robot data, 추가 compute, 학습된 action heads, head LR 변경이 모두 들어 있습니다. 이 비교만으로 Reptile의 optimization rule이 기여했다고 말할 수 없습니다.

이미 `action_reptile_meta_edge2_joint_sft.toml`과 `action_policy_libero_10_edge_sftinit.toml`이 있으므로 새 시스템을 만들 필요는 없습니다. 단, 현재 joint control은 1,000 episodes × 1 update이고 Reptile은 1,000 × 10 updates입니다. **source-SFT가 같은 1,000 episodes를 같은 반복 횟수만큼 학습하는 비교**가 가장 해석하기 쉽습니다. 10,000개의 새 episode를 사용하는 step-matched control은 compute는 맞추지만 서로 다른 data exposure라는 차이가 남습니다. [control TOML:61][control].

검토 당시 source-SFT control 로그의 마지막 기록은 iteration 772였으며, 그에 대응하는 downstream rollout summary는 이번 검색에서 찾지 못했습니다. 이후 진행 상태를 의미하는 것은 아닙니다. 우선 이 대조군을 완료·평가하는 가치가 큽니다. 추가로 기존 meta checkpoint 250/500/750/1000 중 몇 개를 동일 downstream schedule로 비교하면, 58.7시간짜리 meta stage를 끝까지 수행하는 것이 필요한지도 알 수 있습니다.

기본 head LR 5×와 meta-init 1×의 차이도 분리해야 합니다. 최소한 동일 코드에서 fresh-init의 1×/5×, pretrained-init의 1×/5×를 통제하거나, 각 방법을 동일한 제한된 validation budget으로 조정한 결과를 보고하십시오.

6. **DINO 정렬은 정상적으로 동작하지만, ‘DINO를 닮음’과 ‘action이 좋아짐’은 다른 지표입니다.**

확인한 teacher 경로는 native RGB → 각 camera view → future frames 1..16 → 224×224 → DINOv2 post-LayerNorm patch tokens입니다. teacher는 `requires_grad_(False)`, `eval()`, `no_grad()`를 사용합니다. CLS를 제거하며 view/time/space 순서를 유지해 avgpool하고, student의 대응하는 noisy vision token에 연결합니다. 이 recipe의 avgpool target에는 학습 파라미터가 없어 teacher target이 student를 따라 움직이는 collapse 문제도 없습니다. [teacher:124][teacher], [future frames:639][teacher-frames], [target 정렬:195][align].

teacher의 per-view `(16,16,16,768)`을 `(4,5,5,768)`로 줄이고, 두 view를 width로 붙여 sample당 200개 token을 비교합니다. MoT feature는 2048차원이고 약 1,000만 파라미터 MLP를 거쳐 768차원 target과 비교됩니다. 따라서 정렬되는 것은 정확히 **`MLP(h24)`**이지 raw `h24` 자체가 아닙니다. MLP가 얼마나 변환을 담당하고 backbone/action 경로가 얼마나 유용한 정보를 얻는지 별도로 확인해야 합니다. [MLP:110][projector], [vision token 선택:458][vision-select].

실제 v2 validation 마지막 기록은 raw cosine 0.92968, centered cosine 0.89021입니다. 단순한 상수 feature collapse가 지배적이라는 설명은 현재 증거와 잘 맞지 않습니다. 반면 이 높은 정렬 점수가 세밀한 위치·접촉·gripper 제어에 필요한 구별까지 보장하지는 않습니다. 다음 진단이 유용합니다.

- teacher/student의 카메라별·시간별·foreground별 metric, 샘플/공간 위치를 섞은 target과의 비교. 배경과 camera identity만으로 높은 점수가 나오는지 확인합니다.
- backbone에서 `||grad L_action||`, `||grad L_DINO||`, 두 gradient의 cosine을 일부 block에 한해 측정합니다. total loss 비율만으로 간섭의 크기를 판단하지 않습니다.
- projection 전 feature의 frozen probe나 제한된 linear projector 비교. projector만 복잡하게 만드는 것이 정책에 도움이 되는지 확인합니다.
- sigma 구간별 action loss와 REPA loss를 함께 봅니다. 현 recipe는 깨끗한 미래 영상의 feature를 모든 noise level에서 복원하도록 하므로, 관측 특징의 일반화와 미래 영상 복원을 구분해야 합니다. 미래 frame을 **학습 target으로** 쓰는 자체는 leakage가 아닙니다.

24번째 block은 뒤에 4개 block이 있어 action 경로로 정보가 전달될 수 있고, shared parameter에도 gradient가 흐릅니다. ‘action token을 직접 align하지 않으니 전혀 도움이 안 된다’는 주장도 맞지 않습니다. 다만 action에 대한 이득은 간접적입니다. [capture:1038][capture], [action decode:1617][action-decode].

REPA 원 논문에서 앞쪽 block 정렬이 유리했던 것은 이미지 생성 실험의 결과입니다. 여기서 8번째 block이 반드시 더 좋아야 하는 법칙은 아닙니다. 실제 저장 결과도 layer 8은 62.8%, ramp200은 62.0%, weight1+ramp200은 64.4%로 layer 24의 66.2%를 넘지 못했습니다. 따라서 ‘앞쪽 layer로 옮기면 해결’이라고 권하지 않습니다. [REPA 논문](https://arxiv.org/html/2410.06940v2).

7. **pooling의 정보 손실과 meta/post objective 차이는 검증할 가치가 있는 가설입니다.**

4개의 raw frame을 한 temporal cell로, 16×16 patches를 5×5로 평균내므로 짧은 접촉 변화와 작은 물체 위치가 약해질 수 있습니다. teacher target과 VAE token의 시간 bin 개수는 맞지만, VAE의 실제 receptive field가 정확히 네 frame의 비중 없는 평균이라는 뜻은 아닙니다. 이것은 index 오류를 발견했다는 의미가 아니라, alignment target이 가진 근사입니다.

다만 이미 `subgrid122` 결과가 65.8%로 저장되어 있고 기본 v2는 66.2%이므로 pooling을 덜 하는 것만으로 해결된다는 근거는 없습니다. 새 복잡한 adapter보다 task별 실패 영상에서 작은 물체 위치/접촉 문제가 실제로 반복되는지 먼저 확인하는 것이 좋습니다.

meta stage는 `10 L_video + 10 L_action`, downstream은 여기에 `5 L_DINO`가 더해집니다. Reptile이 준비한 adaptation objective와 실제 objective가 달라집니다. 또한 meta는 10 steps/2-step warmup, downstream은 2,000 steps/200-step warmup입니다. 이를 엄밀히 같게 해야만 한다는 뜻은 아니지만, ‘post-training을 그대로 축소했다’는 설명에는 한계가 있습니다. 현재 meta trainer는 raw batch를 넘기지 않아 REPA를 inner loop에서 사용할 수 없다는 점도 코드에 명시되어 있습니다. [training_step_from_inputs:1481][meta-repa-limit].

v2 초기 validation에서 weighted DINO loss는 약 5.02, 전체 loss는 약 9.96이므로 시작부터 큰 새 objective가 붙습니다. 이것이 meta 초기화를 훼손하는지는 gradient 관찰이 필요합니다. 기존 layer8 warmup 실패를 무시한 채 warmup을 일반적인 해결책으로 제안해서는 안 됩니다. 필요하다면 layer24를 고정한 상태에서 작은 projector 사전 적응/weight schedule을 검증하십시오.

8. **현재 validation은 ‘50 demonstrations 전반의 고정 검증’과 다릅니다.**

실제 budget은 512/13,049≈3.92%의 windows입니다. dataset은 episode 순서만 shuffle하고 episode 안에서는 연속 window를 내보냅니다. validation마다 iterator가 시작될 때 epoch=0으로 돌아가며, `in_order=False` worker 도착 순서와 매번 새 diffusion noise의 영향도 받습니다. 따라서 모든 task와 manipulation phase를 균등하게 확인하지 못하고, 동일한 고정 검증 window라고 보장할 수도 없습니다. [iterator:75][iterable], [loader:133][val-loader], [validate:512][validate].

작은 CPU 재현으로 seed123 permutation에서 네 `(rank, worker)`가 처음 만나는 dataset task는 9/6/5/3임을 확인했습니다. 이것은 실제 video decode 속도까지 재현한 coverage 측정은 아니지만, 검증이 task-stratified가 아니라는 사실을 보여줍니다. 실제 task/episode/timestep IDs를 검증 로그에 기록해야 합니다.

v2 action validation loss는 iteration1000에서 0.03945, iteration2000에서 0.11064로 변합니다. 이를 즉시 ‘후반 overfit 확정’이라고 해석하면 안 됩니다. noisy validation을 먼저 고정해야 하기 때문입니다. 권장 방식은 각 task/demo의 진행 구간에서 고정 window를 추출하고 고정 sigma/noise 또는 여러 noise seed를 사용하여 task macro-average를 내는 것입니다. RNG를 저장·복원하거나 전용 generator를 써서 validation 빈도가 학습 trajectory를 바꾸지 않게 하십시오.

비교 시 REPA가 포함된 `val/loss_total`을 REPA 없는 모델의 total loss와 직접 비교하지 말고, 동일 action metric과 rollout SR을 사용해야 합니다. 현재 evaluator와 validation은 EMA를 사용하는 경로가 있으므로 checkpoint 선택/서빙 모두 동일한 EMA 기준으로 기록해야 합니다. EMA warm-start net→EMA 복사는 현행 trainer에 있어 초기 random EMA 문제는 방어되어 있습니다. [EMA copy:304][ema-copy].

9. **3-shot 데이터 예산을 명시해야 합니다.**

30개 train demo와 50개 validation demo가 겹치지 않는 점은 확인했습니다. validation에는 backward가 없어 gradient 학습량을 30개라고 말할 수 있습니다. 다만 validation으로 method/λ/checkpoint를 선택했다면 더 많은 target demonstrations의 정보를 선택 과정에 사용한 것입니다. ‘target 30개로 gradient 학습 + 별도 50개 validation’이라고 쓰거나, target 총 예산이 30개라는 엄격한 설정이면 다른 source development task로 hyperparameter를 정해야 합니다. 단순 모니터링에만 사용했다면 그 점을 명시하면 됩니다.

또한 현재 정규화 통계 metadata에는 LIBERO-10/object/spatial/goal이 모두 기재되어 있습니다. subset에 맞춰 30개에서 재산출하지 않습니다. 이는 379개 전체를 직접 gradient 학습했다는 뜻은 아니지만, 엄격한 ‘30개만 이용’ 조건과는 다릅니다. metadata가 실제 산출 provenance인지 재확인하고, 30개에서 계산한 통계 또는 환경의 알려진 action bounds를 사용하는 비교를 추가하십시오. source pretrained models/DINO의 외부 데이터 사용은 별도 prior로 공개하면 됩니다. [stats JSON:2][stats], [stats loader:240][stats-loader].

정규화를 변경하면 **server의 inverse normalization도 반드시 같은 파일로 변경**해야 합니다. 현재 eval launcher는 기존 bundled stats를 고정 경로로 지정합니다. [eval launcher:32][eval-launcher].

10. **few-shot에서는 window 수보다 batch 다양성이 중요합니다.**

현재 학습 window는 7,618개지만 대부분 인접한 window이고, 16-frame horizon에서 시작점이 한 frame 차이면 입력 clip 17장 중 16장이 겹칩니다. global batch 256을 서로 독립적인 demonstration 256개처럼 해석할 수 없습니다. 현 streaming loader는 worker마다 episode 안을 순차 순회하여 시간 상관이 높은 batch를 만듭니다. [iterator:89][iterable].

task당 demo는 3개로 같아도 유효 window 수는 508–1,217개로 약 2.4배 차이입니다. 모든 window를 균등 사용하는 기준에서는 긴 task가 더 큰 weight를 받지만, 최종 SR은 task마다 동일 비중입니다. task → demonstration → timestep을 균등하게 선택하거나 작은 shuffle buffer를 사용하면 목적과 더 잘 맞습니다. random-access I/O 비용을 피하려면 작은 subset의 VAE latent와 고정 teacher feature를 cache하는 방법이 있습니다. 공간 augmentation을 추가할 때는 teacher/student에 같은 변환을 적용하고, horizontal flip처럼 action/camera geometry를 바꾸는 변환은 일관성 있게 처리해야 합니다.

2,000×256/7,618≈67.2번의 nominal window pass입니다. all379 조건은 95,405 windows라 같은 step budget으로는 약 5.37 pass뿐입니다. 저장된 all379+Reptile SR 53.4%를 ‘데이터가 많으면 오히려 나쁘다’고 해석하기 전에, 동일 compute와 충분한 convergence를 보는 비교를 분리해야 합니다.

11. **수정 우선순위가 높은 구현·관측 항목입니다.**

| 우선순위 | 항목 | 구체적 조치 |
|---|---|---|
| 높음 | query gain의 noise 혼입 | query RNG 분리, pre/post 동일 noise, action/video loss 별도 기록 |
| 높음 | 편중된 partial validation | task/demo/phase별 고정 window manifest, 고정 noise bank, task별 metric |
| 높음 | 평가 provenance 부족 | summary에 checkpoint/EMA/seed/initial states/server info/code revision/normalizer hash 저장 |
| 높음 | 3-shot budget 모호함 | validation·normalizer·pretrained prior의 사용 범위를 실험 표에 명시 |
| 중간 | source-SFT 비교의 update budget 차이 | 동일 source episodes와 exposure 횟수, 동일 downstream recipe로 control |
| 중간 | task/window 불균형 | task/demo-balanced minibatch 및 inner-step 재샘플링 |
| 중간 | DINO loss 해석 부족 | action과 REPA의 gradient norm/cosine, sigma별 지표 |
| 낮음 | 주석·run name 불일치 | K16, 10 steps, 2 GPUs, layer24/28, val512를 정확하게 기록 |
| 낮음 | checkpoint/head 파일 조합 검증 없음 | DCP와 `.pt`의 iteration/base/source/run ID 일치 assertion 추가 |

추가 compute 낭비도 확인했습니다. query는 5회 중 1회만 평가하지만 매 episode에 모두 decode/transfer/VAE encode합니다. 실제 meta stage에서 encode에 합계 19.38시간이 들었습니다. `q_query=8`, support16에서 query의 encode 비중을 1/3로 가정하면 불필요한 4/5회를 건너뛰는 것만으로 약 5.17시간의 encode work를 줄일 여지가 있습니다. 이는 동일 per-window 비용이라는 근사이며 실제 wall-time 절감은 측정해야 합니다. query를 생략할 때도 support sampler의 RNG 순서는 유지해 실험 data가 바뀌지 않게 해야 합니다. [unconditional encode:400][encode].

resume 관련해서는 actual config(`meta_batch_embodiments=1`, outer SGD)에서 당장 오류를 단정할 근거는 없습니다. 다만 `spec_offset=start_iter`는 실패한 episode를 skip하거나 meta-batch>1일 때 소비한 spec 수와 맞지 않을 수 있고, 선택 가능한 outer Adam의 moments는 ReptileMetaState 내부에 있어 별도 checkpoint/resume이 필요합니다. 향후 해당 설정을 사용할 때 보완할 사항입니다. [resume:297][resume], [meta state:151][meta-state].

12. **다음 실험은 다음 순서가 효율적입니다.**

| 순서 | 비교/작업 | 답할 질문 |
|---|---|---|
| 1 | 저장된 모델의 동일 조건 rollout manifest 확보; fixed query/validation 진단 | 관측 차이가 재현되는가? adaptation gain이 실제로 음수인가? |
| 2 | 일반 source-SFT → 동일 30-demo post-training | Reptile rule이 추가 robot pretraining보다 이득인가? |
| 3 | Reptile support를 task-balanced 3-shot episode로, inner minibatch를 재샘플 | target과 맞춘 adaptation 구조가 개선되는가? |
| 4 | stage별 LR/step 수를 제한된 grid로 비교; 기존 meta checkpoints도 활용 | inner adaptation 과도함, 장기 FT의 초기화 소실을 줄일 수 있는가? |
| 5 | 현 v2와 Reptile-only를 같은 seed42 subset에서 복수 training seed로 비교 | +1.8%p가 학습 난수에 견디는가? |
| 6 | 최종 후보만 보조 subset seed 또는 visual perturbation으로 확인 | 고정 30 demos/장면에 특화된 개선인가, 더 넓은 일반화인가? |

사용자가 정한 seed42의 30 demos는 주 benchmark로 계속 고정해도 됩니다. 다른 subset은 최종 주장 범위를 넓히기 위한 보조 검증입니다. 계산 예산이 작다면 새 loss variant 수를 줄이고 source-SFT control 및 training-seed 반복에 먼저 배분하는 편이 현재 증거상 유리합니다. 초기 source mixture의 물리적 차이(3/5/30 FPS, 단팔/양팔, command/FK delta, camera 구성) 때문에 모든 action head를 scratch row31로 공유하는 선택도 source별 gradient 충돌 진단이나 leave-one-source-out으로 확인할 가치가 있습니다. 같은 10-D 형식/quantile scale이 물리적 의미까지 같게 만들지는 않습니다. [source registry:190][source-registry].

검증 범위는 CPU에서 기존 `reptile_meta_test.py`, `episodic_sampler_test.py`, `adapters_test.py`, `alignment_test.py`, `dinov2_teacher_test.py`를 실행한 것으로 **53 passed**입니다. tiny-network update/reset/export, sampler, teacher와 projection/target shape·gradient 등을 검증합니다. distributed FSDP 경로와 실제 Cosmos backward/새 rollout을 검증한 것은 아닙니다. 학습·추론 GPU job은 실행하지 않았습니다.

```bash
cd /gallery_moma/hyunseok.seong/project/cosmos_hs11/packages/cosmos3
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /gallery_moma/hyunseok.seong/anaconda3/envs/cosmos3-pt/bin/python -m pytest \
  --noconftest -o addopts='' -p no:cacheprovider -q \
  cosmos_framework/data/generator/action/meta/reptile_meta_test.py \
  cosmos_framework/data/generator/action/meta/episodic_sampler_test.py \
  cosmos_framework/model/generator/repa/adapters_test.py \
  cosmos_framework/model/generator/repa/alignment_test.py \
  cosmos_framework/model/generator/repa/dinov2_teacher_test.py
```

[meta-toml]: /gallery_moma/hyunseok.seong/project/cosmos_hs11/packages/cosmos3/examples/toml/sft_config/action_reptile_meta_edge2.toml:64
[post-toml]: /gallery_moma/hyunseok.seong/project/cosmos_hs11/packages/cosmos3/examples/toml/sft_config/action_policy_libero_10_edge_reptileinit_repa_dinov2_v2.toml:42
[backbone]: /gallery_moma/hyunseok.seong/project/cosmos_hs11/packages/cosmos3/cosmos_framework/model/generator/reasoner/nemotron_3_dense_vl/configs/Nemotron-2B-Dense-VL.json:6
[theta-keys]: /gallery_moma/hyunseok.seong/project/cosmos_hs11/packages/cosmos3/cosmos_framework/configs/base/experiment/action/meta/action_reptile_meta_edge.py:38
[eval-summary]: /gallery_moma/hyunseok.seong/project/cosmos_hs11/packages/cosmos3/cosmos_framework/simulation/libero/closed_loop_eval.py:1317
[meta-log]: /gallery_moma/hyunseok.seong/project/cosmos_storage/outputs/cosmos3_action_meta/reptile_meta/edge_reptile_full_k8w16_s10_eps0.5_seed42_v2/reptile_train_log.jsonl:1
[meta-query]: /gallery_moma/hyunseok.seong/project/cosmos_hs11/packages/cosmos3/cosmos_framework/scripts/train_action_reptile.py:373
[noise]: /gallery_moma/hyunseok.seong/project/cosmos_hs11/packages/cosmos3/cosmos_framework/model/generator/omni_mot_model.py:1497
[source-transform]: /gallery_moma/hyunseok.seong/project/cosmos_hs11/packages/cosmos3/cosmos_framework/configs/base/experiment/action/meta/action_reptile_meta_edge.py:178
[interpolation]: /gallery_moma/hyunseok.seong/project/cosmos_hs11/packages/cosmos3/cosmos_framework/data/generator/action/meta/reptile_meta.py:231
[reset]: /gallery_moma/hyunseok.seong/project/cosmos_hs11/packages/cosmos3/cosmos_framework/data/generator/action/meta/reptile_meta.py:103
[sampler]: /gallery_moma/hyunseok.seong/project/cosmos_hs11/packages/cosmos3/cosmos_framework/data/generator/action/meta/episodic_sampler.py:205
[inner-loop]: /gallery_moma/hyunseok.seong/project/cosmos_hs11/packages/cosmos3/cosmos_framework/scripts/train_action_reptile.py:424
[packing]: /gallery_moma/hyunseok.seong/project/cosmos_hs11/packages/cosmos3/cosmos_framework/data/generator/action/meta/episodic_sampler.py:550
[control]: /gallery_moma/hyunseok.seong/project/cosmos_hs11/packages/cosmos3/examples/toml/sft_config/action_reptile_meta_edge2_joint_sft.toml:61
[teacher]: /gallery_moma/hyunseok.seong/project/cosmos_hs11/packages/cosmos3/cosmos_framework/model/generator/repa/dinov2_teacher.py:124
[teacher-frames]: /gallery_moma/hyunseok.seong/project/cosmos_hs11/packages/cosmos3/cosmos_framework/model/generator/omni_mot_model.py:639
[align]: /gallery_moma/hyunseok.seong/project/cosmos_hs11/packages/cosmos3/cosmos_framework/model/generator/repa/alignment.py:195
[projector]: /gallery_moma/hyunseok.seong/project/cosmos_hs11/packages/cosmos3/cosmos_framework/model/generator/repa/adapters.py:110
[vision-select]: /gallery_moma/hyunseok.seong/project/cosmos_hs11/packages/cosmos3/cosmos_framework/model/generator/mot/cosmos3_vfm_network.py:458
[capture]: /gallery_moma/hyunseok.seong/project/cosmos_hs11/packages/cosmos3/cosmos_framework/model/generator/mot/unified_mot.py:1038
[action-decode]: /gallery_moma/hyunseok.seong/project/cosmos_hs11/packages/cosmos3/cosmos_framework/model/generator/mot/cosmos3_vfm_network.py:1617
[meta-repa-limit]: /gallery_moma/hyunseok.seong/project/cosmos_hs11/packages/cosmos3/cosmos_framework/model/generator/omni_mot_model.py:1481
[iterable]: /gallery_moma/hyunseok.seong/project/cosmos_hs11/packages/cosmos3/cosmos_framework/data/generator/action/datasets/action_sft_dataset.py:75
[val-loader]: /gallery_moma/hyunseok.seong/project/cosmos_hs11/packages/cosmos3/cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_libero_edge_repa.py:133
[validate]: /gallery_moma/hyunseok.seong/project/cosmos_hs11/packages/cosmos3/cosmos_framework/trainer/__init__.py:512
[ema-copy]: /gallery_moma/hyunseok.seong/project/cosmos_hs11/packages/cosmos3/cosmos_framework/trainer/__init__.py:304
[stats]: /gallery_moma/hyunseok.seong/project/cosmos_hs11/packages/cosmos3/cosmos_framework/data/generator/action/normalizer_stats/libero_native_frame_wise_relative_rot6d.json:2
[stats-loader]: /gallery_moma/hyunseok.seong/project/cosmos_hs11/packages/cosmos3/cosmos_framework/data/generator/action/datasets/libero_lerobot_dataset.py:240
[eval-launcher]: /gallery_moma/hyunseok.seong/project/cosmos_hs11/packages/cosmos3/examples/eval_libero_closed_loop.sh:32
[encode]: /gallery_moma/hyunseok.seong/project/cosmos_hs11/packages/cosmos3/cosmos_framework/scripts/train_action_reptile.py:400
[resume]: /gallery_moma/hyunseok.seong/project/cosmos_hs11/packages/cosmos3/cosmos_framework/scripts/train_action_reptile.py:297
[meta-state]: /gallery_moma/hyunseok.seong/project/cosmos_hs11/packages/cosmos3/cosmos_framework/data/generator/action/meta/reptile_meta.py:151
[source-registry]: /gallery_moma/hyunseok.seong/project/cosmos_hs11/packages/cosmos3/cosmos_framework/data/generator/action/meta/embodiments.py:190
