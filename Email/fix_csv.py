import csv
import os
import shutil

csv_path = os.path.join('spam/')

rows_train= []
rows_test = []
pathologies = []
with open(csv_path+'spam.csv', 'r') as csvfile:
     spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
     cont = 0
     for row in spamreader:
        del row[2:]
        if cont==0:
            row[0] = "Label"
            row[1] = "Mail"
            rows_train.append(row)
            rows_test.append(row)
        else:
            if cont>4738:
                rows_test.append(row)
            else:
                rows_train.append(row)
        cont += 1

file2 = open('spam_train.csv', 'w', newline='')
writer = csv.writer(file2)
writer.writerows(rows_train)
file2.close()

file3 = open('spam_test.csv', 'w', newline='')
writer = csv.writer(file3)
writer.writerows(rows_test)
file3.close()
