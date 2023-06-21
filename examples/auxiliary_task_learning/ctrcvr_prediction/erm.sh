# Single Task Learning
for domain_name in NL ES FR US ; do
  CUDA_VISIBLE_DEVICES=0 python erm.py -tr click -ts click \
    --domain_names ${domain_name} --log logs/STL_click/${domain_name}
done

for domain_name in NL ES FR US ; do
  CUDA_VISIBLE_DEVICES=0 python erm.py -tr conversion -ts conversion \
    --domain_names ${domain_name} --log logs/STL_conversion/${domain_name}
done

# Equal Weight with Shared Embedding Model
for domain_name in NL ES FR US ; do
  CUDA_VISIBLE_DEVICES=0 python erm.py -tr click conversion -ts click conversion \
    --domain_names ${domain_name} --log logs/EW/${domain_name}
done

# Equal Weight with MMoE Model
for domain_name in NL ES FR US ; do
  CUDA_VISIBLE_DEVICES=2 python erm.py -tr click conversion -ts click conversion \
    --domain_names ${domain_name} --log logs/EW_mmoe/${domain_name} --model_name mmoe
done
