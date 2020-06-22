
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

with open('Result.csv', 'r') as t1, open('Test.csv', 'r') as t2:
    fileone = t1.readlines()
    filetwo = t2.readlines()

total=10
counter=0
i=1;
a=0;
while i<190:
    if(fileone[i]==filetwo[i]):
        a=a+1
    i=i+1
print(" accuracy= "+str(a/190))

Y_True= []
for i in range(190):
    Y_True.append(ord(filetwo[i][-2])-48)


Y_Predicted= []
for i in range(190):
    Y_Predicted.append(ord(fileone[i][-2])-48)

del Y_True[0]
del Y_Predicted[0]

target_names = ['Dog', 'Cat']
print(classification_report(Y_True, Y_Predicted,target_names=target_names))
print("confusion_matrix")
print(confusion_matrix(Y_True, Y_Predicted))

