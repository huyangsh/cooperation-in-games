# AER paper.
python ./monopoly.py --player_type 0 --T 2000000 --log_freq 50000 --seed
python ./monopoly.py --player_type 1 --T 10000000 --log_freq 200000 --seed
python ./monopoly.py --player_type 1 --T 100000000 --log_freq 200000 --alpha 0.4 --beta 2e-7 --seed