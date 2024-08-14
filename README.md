# 준비

- MVTEC dataset 을 다운받아 path a 에 저장
- `main.py` 내 `mvtec_data_path` 를 path a 로 지정

# 실행 예제

- `python main.py --wan` 으로 wandb 끄고 실행. wandb 사용하려면 wandb.init() 부분에서 wandb id 변경하고 실행.
- `python main.py --wan --subdatasets capsule` mvtec capsule 에 대해 실험 진행
- `python main.py --wan --subdatasets capsule+pill` mvtec capsule+pill 에 대해 실험 진행(capsule 과 pill 을 자동 병합함)
- `python main.py --wan --mainmodel simple` SimpleNet 으로 실험진행
- `python main.py --wan --mainmodel patchcore` PatchCore 로 실험진행
- `python main.py --wan --mainmodel vig_score_patchlevel` VIG 를 사용하는 우리 방법의 초기 버전으로 실험진행(최고 성능이 나오는 버전은 아니며, 최적의 hyperparameter 를 찾기 위해 wandb sweep 필요)

# 실행 환경

- 현재 환경은 아래와 같지만, 동일한 환경을 구성해야 코드가 동작하는 것은 아니기 때문에, 실행이 불가한 library 에 대해서만 설치하는 것을 추천
- Ubuntu 20.04, CUDA 11.7, python 3.9.17
- 나머지 `requirements.txt` 참조
