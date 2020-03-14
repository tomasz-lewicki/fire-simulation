time python3 ../example.py --no-png-output >> "$(lscpu | grep "Model name:" | sed -r 's/Model name:\s{1,}//g')".txt
