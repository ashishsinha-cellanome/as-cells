import os, json, glob

phase2_dir = '/mnt/direct-attached/PHASE2'
standard_categories = {
    0: "cell",
    1: "bead",
    2: "cell-adhered",
    3: "soma"
}

count = 0
for json_path in glob.glob(os.path.join(phase2_dir, '**', '*.json'), recursive=True):
    if not json_path.endswith('_annotations.json') and not json_path.endswith('merged_val_annotations.json'):
        continue
    
    with open(json_path, 'r') as f:
        try:
            data = json.load(f)
        except Exception:
            continue
            
    if 'categories' in data:
        updated = False
        for cat in data['categories']:
            cat_id = cat.get('id')
            if cat_id in standard_categories and cat.get('name') != standard_categories[cat_id]:
                cat['name'] = standard_categories[cat_id]
                updated = True
        
        if updated:
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Fixed categories list in: {json_path}")
            count += 1

print(f"Done. Fixed {count} files.")
