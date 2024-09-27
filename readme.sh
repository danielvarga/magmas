mv equations.txt original_equations.txt
cat original_equations.txt | sed "s/◦︎/+/g" > equations.txt


cat equations.txt | grep -v "[uv]" > equations_4.txt
cat equations_4.txt | tr '=' '\n' | sed "s/^ *//" | sed "s/ *$//" | sort | uniq > formulae_4.txt

cat equations_4.txt | head -20 | python main.py
