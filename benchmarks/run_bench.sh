time python3 ../example.py >> "$(lscpu | grep "Model name:" | sed -r 's/Model name:\s{1,}//g')".txt
