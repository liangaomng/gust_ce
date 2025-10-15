
#Forward simulation data--section3.1
python Train/train.py --yaml_path "Bash/Command/Simu_section3_1.yaml"
#Forward exprimental data--section3.2
python Train/train.py --yaml_path "Bash/Command/Expr_section3_2.yaml"
#Innverse exprimental data--section3.3
python Train/train.py --yaml_path "Bash/Command/Expr_section3_3.yaml"


##hard
python Train/train.py --yaml_path "Bash/Command/Expr_section3_2_hard.yaml"

#hard x-only 1case for test
python Train/train.py --yaml_path "Bash/Command/Expr_section3_x.yaml"


#2 type graph adj--downstream
python Train/train.py --yaml_path "Bash/Command/Expr_section4_1.yaml" # downstream
python Train/train.py --yaml_path "Bash/Command/Expr_section4_2.yaml"