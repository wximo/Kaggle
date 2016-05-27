import csv

def loadCsv(path):
    reader=csv.reader(file(path,'rb'))
    l=[]
    for line in reader:
        l.append(line)
    l.remove(l[0])

    return l

def putCsv(path,data):
    writer=csv.writer(file(path,'wb'))
    for line in data:
        writer.writerow(line)




if __name__=='__main__':
    train_path="/Users/Ximo/Documents/Kaggle/Digit_Recognizer/train.csv"
    test_path="/Users/Ximo/Documents/Kaggle/Digit_Recognizer/test.csv"
    save_path="/Users/Ximo/Documents/Kaggle/Digit_Recognizer/result.csv"

    train_data=loadCsv(train_path)

    train_data_x=train_data[:,1:]
    train_data_y=train_data[:,0]
    
    test_data=loadCsv(test_path)

    


    
