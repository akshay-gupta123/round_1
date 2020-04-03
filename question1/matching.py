import json

def official(name):
    space = 0
    lower = 0
    special = 0
    number = 0


    for i in range(len(name)):
        if (ord(name[i])>=48)and(ord(name[i])<=57):
            number+=1
        elif (ord(name[i]) >= 97)and(ord(name[i]) <= 122):
            lower+=1
        elif(ord(name[i])==32):
            space+=1
        elif not((ord(name[i])>=65) and (ord(name[i])<=90)):
            special+=1

    if( special or  number):
        return 0
    elif(not space):
        return 0
    elif((lower-space)==len(name)):
        return 0
    else:
        return 1

csv_location = input("enter the location of csv file")

official_name = []
project = []
organization = []

with open(csv_location,encoding='utf-8') as f:
    for line in f:
        index1 = line.index(',')
        name = line[0:index1]
        index2 = line[index1+1:].index(',')
        org = line[index1+1:index1+index2+1]
        pro = line[index1+index2+2:]
        if not official(name):
            print(name)
        else:
          official_name.append(name)
          project.append(pro)
          organization.append(org)


f = open('students.json')
data = json.load(f)

for i in data:
     for j in range(len(official_name)):
       if official_name[j] == i['n']:
          print(official_name[j]+","+i['i']+","+i['d']+','+organization[j]+','+project[j])
          break