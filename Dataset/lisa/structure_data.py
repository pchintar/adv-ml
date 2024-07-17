import os
import shutil

old_dir = 'old'
new_dir = 'train'

try:
    items = os.listdir(old_dir)
except WindowsError:
    print('old file missing')
    exit()
    
for item in items:
    name_count = 0
    folder = ""
    if item[-3:] == 'png':
        print(item)

        for l in item:
            if(l == '_'):
                break
            else:
                folder += l
        exist = os.path.exists(new_dir + '/' + str(folder))
        if not exist:
            os.makedirs(new_dir + '/' + str(folder))
        
        shutil.move(old_dir + '/' + str(item), new_dir + '/' + str(folder))

print('done')