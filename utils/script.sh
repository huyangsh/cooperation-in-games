# AER paper.
python ./monopoly.py --player_type 0 --T 2000000 --log_freq 50000 --seed

python ./monopoly.py --player_type 1 --T 100000000 --log_freq 1000000 --T_eval 100 --seed
python ./monopoly.py --player_type 1 --T 100000000 --log_freq 200000 --alpha 0.4 --beta 2e-7 --T_eval 100 --seed

python ./monopoly.py --player_type 2 --T 100000000 --log_freq 1000000 --T_eval 100 --seed

python ./monopoly.py --player_type 1 --T 100000000 --log_freq 1000000 --T_eval 100 --runner oracle --seed