# Cosmos hs12 — masked visual feature prediction

`cosmos_hs10`의 작업 디렉터리 소스를 기반으로 만든 LIBERO-10 few-shot post-training 실험입니다.
원본 hs10은 수정하지 않았으며 학습 출력/캐시는 복사하지 않았습니다.

기존 action/video FM loss에, pixel masking → frozen VAE → MoT block 8 → 작은 projection → frozen V-JEPA 2.1
teacher target 회귀를 추가합니다. 보조 입력에 action/caption을 넣지 않으며 conditioning frame도 마스킹합니다.
Teacher target은 고정된 avgpool과 channel normalization을 거치고 gradient를 받지 않습니다.

- [설계, 실행 방법, 평가 조건과 검증 범위](packages/cosmos3/docs/action_policy_libero_masked_jepa.md)
- [실행 스크립트](packages/cosmos3/examples/launch_sft_action_policy_libero_10_edge_masked_jepa.sh)
- [기본 dense recipe](packages/cosmos3/examples/toml/sft_config/action_policy_libero_10_edge_masked_jepa.toml)
- [마스킹과 보조 loss 코드](packages/cosmos3/cosmos_framework/model/generator/repa/masked_prediction.py)

`JEPA_VARIANT=dense|masked_only|linear`로 세 버전을 선택합니다. 고정된 30 training demos와 겹치지 않는
50 validation demos를 사용합니다. 기존 bundled action normalization 통계의 사용 범위는 위 문서에 명시했습니다.

CPU 단위/통합 테스트와 세 recipe의 training dryrun을 검증했습니다. 현재 환경에 GPU가 없어 실제 checkpoint로
수행하는 FSDP 학습과 LIBERO rollout 성능은 검증하지 못했습니다. 성능 개선을 관측한 결과물은 아닙니다.
