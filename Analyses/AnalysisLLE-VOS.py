import os


if __name__ == '__main__':

    directory = '/Users/giuliadangelo/workspace/data/DATASETs/EVIMO2LowightChallengingConditions/'
    attention_dir = directory + 'attention/'


    dirs_att = [d for d in os.listdir(attention_dir) if os.path.isdir(os.path.join(attention_dir, d))]
    dirs_att = sorted(dirs_att)
    accuracies = []
    for dir in dirs_att:
        txt_files = [f for f in os.listdir(attention_dir + dir) if f.endswith('.txt')]
        #load the txt file and print the value written in it
        for txt_file in txt_files:
            with open(attention_dir + dir + '/' + txt_file, 'r') as f:
                text = f.read()
                acc = round(float(text.split(':')[1].strip()),2)
                accuracies.append(acc)
                print(text)
    print('Mean accuracy:', round(sum(accuracies)/len(accuracies),2))
