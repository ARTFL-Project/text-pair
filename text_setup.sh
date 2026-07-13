# Clean any prior input
rm -rf in-and-out/eltec-100_held/*
mkdir -p in-and-out/eltec-100_held

# Flatten the splits into TextPAIR's expected location.
# `find ... -exec cp` walks all subdirectories and copies every .xml file
# to the flat staging directory.
find ../sextant/projects/eltec-100/splits/ -name "*" -exec cp {} in-and-out/eltec-100_held/ \;

# Sanity check: how many files landed?
ls in-and-out/eltec-100_held/ | wc -l

sed -i '' 's|^source_file_path =.*|source_file_path = /Users/sextant2/Repos/text-pair/in-and-out/eltec-100_held|' my_config.ini

export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - zsh)"

textpair --config=my_config.ini \
         --skip_web_app \
         --output_path=/tmp/textpair-eltec-100-out \
         --workers=1 \
         eltec-100_thesis


mv ../sextant/projects/eltec-100/alignments/alignments.jsonl \
   ../sextant/projects/eltec-100/alignments/alignments.jsonl.bak.$(date +%Y%m%d-%H%M%S) \
   2>/dev/null

mkdir -p ../sextant/projects/eltec-100/alignments

lz4 -d /tmp/textpair-eltec-100-out/results/alignments.jsonl.lz4 \
       ../sextant/projects/eltec-100/alignments/alignments.jsonl
