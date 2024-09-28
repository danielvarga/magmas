mv equations.txt original_equations.txt
cat original_equations.txt | sed "s/◦︎/+/g" > equations.txt


cat equations.txt | grep -v "[uv]" > equations_4.txt
cat equations_4.txt | tr '=' '\n' | sed "s/^ *//" | sed "s/ *$//" | sort | uniq > formulae_4.txt

cat equations_4.txt | head -20 | python main.py

# or maybe just equations built only from x, 31 equations:
cat equations_4.txt | grep -v "[yzw]" | python main.py
# or from x and y. there are 810, in 173 equivalence classes for 2-magmas:
cat equations_4.txt | grep -v "[zw]" | python main.py
cat tao_examples.txt | grep -v "u" | python main.py

