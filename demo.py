from tkinter import *
from tkinter import ttk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from pandastable import Table, TableModel
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, confusion_matrix
from sklearn.ensemble import VotingClassifier
from sklearn.tree import export_text
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import preprocessing as pp

path = "income.csv" 
read_data = pd.read_csv(path)
#Event Function:
class DataCleaning(BaseEstimator, TransformerMixin):
            # def __init__(self):
            def fit(self, X, y):
                return self
            def transform(self, X, y):
                X_ = X.copy()
                y_ = y.copy()
                #Bỏ cột dữ liệu educat.ional-num
                X_ = X_.drop(['educational-num'], axis = 1)
                # Xử lý thuộc tính workclass có giá trị NaN thành giá trị xuất hiện nhiều nhất 
                X_['workclass'].fillna(X_.workclass.mode().to_string(), inplace=True)
                # Xử lý thuộc tính occupation, native-country có giá trị NaN thành giá trị xuất hiện nhiều nhất 
                X_ = X_.fillna(value={'occupation':'Craft-repair','native-country':'United-States'})
                return X_, y_
            def fit_transform(self, X, y):
                return self.fit(X, y).transform(X, y)
class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
            def __init__(self,columns=None):
                self.columns = columns

            def fit(self,X, y=None):
                return self

            def transform(self, X, y=None):
                X_ = X# .copy()
                if self.columns is not None:
                    for col in self.columns:
                        X_[col] = LabelEncoder().fit_transform(X_[col])
                else:
                    for colname,col in X_.iteritems():
                        X_[colname] = LabelEncoder().fit_transform(col)
                return X_
            def fit_transform(self,X,y=None):
                return self.fit(X,y).transform(X)
class GUI():
    def __init__(self, read_data):
        self.read_data = read_data

        self.window = Tk()
        self.window.title("Data Mining")
        self.window.geometry("1620x980")
        self.style = ttk.Style()
        self.style.configure('Blue.TLabelframe.Label', font=('courier', 25, 'bold'))
        self.style.configure('Blue.TLabelframe.Label', foreground ='blue')

        #First Part:  "Preprocessing data"
        self.labelframePre = ttk.LabelFrame(self.window, text="Tiền xử lý dữ liệu", style="Blue.TLabelframe")
        self.labelframePre.pack(fill="both")

        self.frame_displaydata=Frame(self.labelframePre,width=40,height=1)
        self.frame_displaydata.pack(fill=BOTH,expand=0)
        self.btnDisplay = Button(self.labelframePre, text="Hiển thị dữ liệu", command=self.displaydata,font='Helvetica 15 bold')
        self.btnDisplay.pack(side=TOP, padx=5, pady=5, fill="both")

        self.frame_processing=Frame(self.labelframePre,width=40,height=1)
        self.frame_processing.pack(fill=BOTH,expand=0)
        self.btnDisplay = Button(self.labelframePre, text="Xử lý dữ liệu", command=self.balancedata,font='Helvetica 15 bold')
        self.btnDisplay.pack(side=TOP, padx=5, pady=5, fill="both")

        #Second Part: "Extract Feauture"
        self.labelframeExt = ttk.LabelFrame(self.window, text="Chia dữ liệu", style="Blue.TLabelframe")
        self.labelframeExt.pack(fill="both")
        

        self.frame_split = Frame(self.labelframePre,width=40, height=1)
        self.frame_split.pack(fill=BOTH,expand=0)
        self.btnDisplay= Button(self.labelframeExt, text="Dữ liệu train", command=self.data_train,font='Helvetica 15 bold')
        self.btnDisplay.pack(side=TOP, padx=5, pady=5, fill="both")

        self.frame_split_test = Frame(self.labelframePre,width=40, height=1)
        self.frame_split_test.pack(fill=BOTH,expand=0)
        self.btnDisplay = Button(self.labelframeExt, text="Dữ liệu test", command=self.data_test,font='Helvetica 15 bold')
        self.btnDisplay.pack(side=TOP, padx=5, pady=5, fill="both")
        self.lbtemp2 = Label(self.labelframeExt, text="Loại đặc trưng", font='Helvetica 20 bold')
        self.lbtemp2.pack(side= LEFT, padx=7, pady=0)
        self.var = IntVar()
        self.R1 = Radiobutton(self.labelframeExt, text="PCA", variable=self.var, value=1,font='Helvetica 15 bold')
        self.R1.pack( anchor = W, side= LEFT, padx=5, pady=0)

        self.R2 = Radiobutton(self.labelframeExt, text="OHD", variable=self.var, value=2,font='Helvetica 15 bold')
        self.R2.pack( anchor = W, side= LEFT, padx=5, pady=0)
        
        self.R3 = Radiobutton(self.labelframeExt, text="None", variable=self.var, value=3,font='Helvetica 15 bold')
        self.R3.pack( anchor = W, side= LEFT, padx=5, pady=0)

        # self.btnExt = Button(self.labelframeExt, text="Rút trích", command=self.extractdata,font='Helvetica 10 bold')
        # self.btnExt.pack(side=BOTTOM, padx=5, pady=5, fill="both")

        # self.frame1 = Frame(self.window)
        # self.lbtemp3 = Label(self.frame1, text="Dữ liệu train", font='Helvetica 10 bold')
        # self.lbtemp3.pack(side=LEFT)
        # self.lbtemp4 = Label(self.frame1, text="Dữ liệu test", font='Helvetica 10 bold')
        # self.lbtemp4.pack(side=RIGHT)
        # self.frame1.pack(fill="both")
        # self.frame2 = Frame(self.window)
        # self.listboxtrain = Listbox(self.frame2, width=160, height=6)
        # self.listboxtrain.pack(side=LEFT, padx=5, pady=0)
        # self.listboxtest = Listbox(self.frame2, width=160, height=6)
        # self.listboxtest.pack(side=RIGHT, padx=5, pady=0)
        # self.frame3 = Frame(self.window)
        # self.lbtemp4 = Label(self.frame3)
        # self.lbtemp4.pack(side=LEFT)
        # # self.listboxtime = Listbox(self.frame3, width=40, height=2)
        # #self.listboxtime.pack(side=LEFT, padx=5, pady=0)
        # self.frame2.pack(fill="both")
        # self.frame3.pack(fill="both")


        labelframeTrain = ttk.LabelFrame(self.window, text="Huấn luyện và kiểm tra", style="Blue.TLabelframe")
        labelframeTrain.pack(fill="both")
        # self.lbtemp = Label(labelframeTrain, text="Đồ thị hỗ trợ - Đồ thị mục tiêu: ", font='Helvetica 10 bold')
        # self.lbtemp.pack(side= LEFT, padx=5, pady=0)

        # self.comboboxtrain = ttk.Combobox(labelframeTrain)
        # self.comboboxtrain.pack(side=LEFT, padx=5, pady=0)

        self.btnTrain = Button(labelframeTrain, text="Train", command=self.traindata,font='Helvetica 15 bold')
        self.btnTrain.pack(side=RIGHT, padx=5, pady=5, fill="both")


        self.frame4 = Frame(self.window)
        self.lbtemp = Label(self.frame4, text="Phương pháp thực nghiệm: ", font='Helvetica 15 bold')
        self.lbtemp.pack(side= LEFT, padx=5, pady=0)

        self.comboboxmethod = ttk.Combobox(self.frame4)
        self.comboboxmethod.pack(side=LEFT, padx=5, pady=0)
        self.comboboxmethod['value']=('LogisticRegression','DecisionTreeClassifier','SupportVectorMachine','RandomForestClassifier')
        self.btnChart = Button(self.frame4, text="Chart", command=self.chart,font='Helvetica 15 bold')
        self.btnChart.pack(side=BOTTOM, padx=5, pady=5, fill="both")

        self.frame5 = Frame(self.window)
        self.lbtemp = Label(self.frame5, text="Chi tiết cài đặt ", font='Helvetica 20 bold')
        self.lbtemp.pack(side= LEFT, padx=5, pady=0)

        self.frame6 = Frame(self.window)
        self.lbtemp = Label(self.frame6, text="Loại đặc trưng ", font='Helvetica 15 ')
        self.lbtemp.pack(side= LEFT, padx=5, pady=0)
        self.listboxmethod = Listbox(self.frame6, width=40, height=1,font=('Helvetica', 14))
        self.listboxmethod.pack(fill=BOTH)

        self.frame7 = Frame(self.window)
        self.lbtemp = Label(self.frame7, text="Phương pháp máy học ", font='Helvetica 15 ')
        self.lbtemp.pack(side= LEFT, padx=5, pady=0)
        self.listboxmethodMC = Listbox(self.frame7, width=40, height=1,font=('Helvetica', 14))
        self.listboxmethodMC.pack(fill=BOTH)

        self.frame8 = Frame(self.window)
        self.lbtemp = Label(self.frame8, text="Kết quả thực nghiệm ", font='Helvetica 20 bold')
        self.lbtemp.pack(side= LEFT, padx=5, pady=0)

        self.frame9 = Frame(self.window)
        self.lbtemp = Label(self.frame9, text="Độ chính xác ", font='Helvetica 15 ')
        self.lbtemp.pack(side= LEFT, padx=5, pady=0)
        self.listboxE = Listbox(self.frame9, width=40, height=1, font=('Helvetica', 14))
        self.listboxE.pack(fill=BOTH)
        self.frame10 = Frame(self.window)
        # self.lbtemp = Label(self.frame10, text="Thời gian ", font='Helvetica 10 ')
        # self.lbtemp.pack(side= LEFT, padx=5, pady=0)
        # self.listboxETime = Listbox(self.frame10, width=40, height=1)
        # self.listboxETime.pack(side=LEFT, padx=5, pady=0)
        #self.frame6 = ttk.LabelFrame(self.window, text="Chi tiết cài đặt")
        #self.frame7 = ttk.LabelFrame(self.window, text="Kết quả thực nghiệm")
        self.frame4.pack(fill="both")
        self.frame5.pack(fill="both")
        self.frame6.pack(fill="both")
        self.frame7.pack(fill="both")
        self.frame8.pack(fill="both")
        self.frame9.pack(fill="both")
        self.frame10.pack(fill="both")
        self.window.mainloop()

    def displaydata(self):
        self.current = 1
        temp = TableModel(dataframe=self.read_data)
        self.table = pt = Table(self.frame_displaydata, model=temp,
                                 showstatusbar=True)
        pt.show()
        ################################
    def preprocessor(self,X, y):
        X, y = DataCleaning().fit_transform(X, y)
        X = MultiColumnLabelEncoder().fit_transform(X)
        column_name = X.columns
        indices = X.index
        X = StandardScaler().fit_transform(X)
        return pd.DataFrame(X, columns=column_name, index=indices), y
    def balancedata(self):
        
        X = self.read_data.drop(columns ="income_>50K")
        y = self.read_data["income_>50K"]
        X, y = DataCleaning().fit_transform(X, y)
        cat_features = X.select_dtypes(exclude=[np.number]).columns.to_list()
        MCLE = MultiColumnLabelEncoder(cat_features)
        X = MultiColumnLabelEncoder(cat_features).fit_transform(X)
        X = StandardScaler().fit_transform(X)
        
        X = self.read_data.drop("income_>50K", axis=1)
        y = self.read_data["income_>50K"].copy()
        X, y = self.preprocessor(X, y)
        #Gộp lại như cũ
        self.read_data = pd.concat([X, y], axis=1, sort=False)
        if self.current==1:
            self.frame_displaydata.forget()
        self.current = 2
        temp = TableModel(dataframe=self.read_data)
        self.table = pt = Table(self.frame_processing, model=temp,
                                 showstatusbar=True)
        pt.show()
        
        ######################################
    def processing_split(self):

        data_X_0 = self.read_data[self.read_data['income_>50K'] ==0]
        data_Y_0 = self.read_data[self.read_data['income_>50K'] ==1]

        train_data_0, test_data_0 = train_test_split(data_X_0, test_size = 0.3, random_state = 42)
        train_data_1, test_data_1 = train_test_split(data_Y_0, test_size = 0.3, random_state = 42)

        train_data = pd.concat([train_data_0, train_data_1])
        train_data = train_data.sample(n = train_data.shape[0], random_state = 42)
        test_data = pd.concat([test_data_0,test_data_1])
        test_data = test_data.sample(n = test_data.shape[0] ,random_state = 42)
        train_data.shape
        #Tập dữ liệu train
        self.X_train = train_data.drop("income_>50K", axis=1)
        self.y_train = train_data["income_>50K"].copy()
        #Tập dữ liệu test
        self.X_test = test_data.drop("income_>50K", axis=1)
        self.y_test = test_data["income_>50K"].copy()

    def data_train(self):
        self.processing_split()
        if self.current == 2:
            self.frame_processing.forget()
        self.current = 3
        temp = TableModel(dataframe=self.X_train)
        self.table = pt = Table(self.frame_split, model=temp,
                                 showstatusbar=True)
        pt.show()
        
    def data_test(self):
        self.processing_split()
        if self.current == 3:
            self.frame_split.forget()
        self.current=4
        temp = TableModel(dataframe=self.X_test)
        self.table = pt = Table(self.frame_split_test, model=temp,
                                 showstatusbar=True)
        pt.show()
    def traindata(self):
        tempName = ""
        if self.var.get() == 1:
            tempName = "PCA"
        elif self.var.get() == 2:
            tempName = "OHD"
        elif self.var.get() == 3: 
            tempName ="None"
        self.listboxmethod.insert(0,tempName)
        self.listboxmethodMC.insert(0,self.comboboxmethod.get())
        method = self.comboboxmethod.get()
        #chọn mẹ gì đó thì ở đây lấy ra giá trị được chọn(combobox)
        #print(self.comboboxmethod.get())--> decisionTreeClassi=fer
        #Kiểu Radio buttton(none,LCA,...) var.get() -> 1  or 2 or 3
        self.processing_split()
        """None"""
        """LogisticRegression"""
        def LR(X_train,y_train):
            LR_model = LogisticRegression(C = 100, solver='liblinear', random_state=0, max_iter=1000)
            LR_model.fit(X_train, y_train)
            y_pred = LR_model.predict(self.X_test) #Kết quả dự đoán tập test

            y_train_pred = cross_val_predict(LR_model, X_train, y_train, cv=3)
            accuracy_score_LR = accuracy_score(y_train, y_train_pred)
            return accuracy_score_LR
        self.None_LR = LR(self.X_train,self.y_train)
        if tempName=="None" and method == "LogisticRegression":
            self.listboxE.insert(0,self.None_LR)
        
        """DecisionTreeClassifier"""
        def DT(X_train,y_train):
            X_train_name = [var for var in self.X_train.columns]
            DT_model = DecisionTreeClassifier(random_state=0, max_depth=3)
            DT_model = DT_model.fit(self.X_train, self.y_train)
            y_train_str = self.y_train.astype(str)
            y_train_pred = cross_val_predict(DT_model, self.X_train, self.y_train, cv=3)
            accuracy_score_DT = accuracy_score(self.y_train, y_train_pred)
            return accuracy_score_DT
        self.None_DT = DT(self.X_train,self.y_train)
        if tempName=="None" and method == "DecisionTreeClassifier":
            self.listboxE.insert(0,self.None_DT)
        
        """SVM"""
        def SVM(X_train,y_train):
            SVM_Linear_model = Pipeline([
                ("scaler", StandardScaler()),
                ("linear_svc", LinearSVC(C=10, loss="hinge", max_iter = 1000000)),
                ])
            SVM_Linear_model.fit(X_train, y_train)
            y_train_pred = cross_val_predict(SVM_Linear_model, X_train, y_train, cv=3)
            accuracy_score_SVM_Linear = accuracy_score(y_train, y_train_pred)
            return accuracy_score_SVM_Linear
        self.None_SVM = SVM(self.X_train,self.y_train)
        if tempName=="None" and method == "SupportVectorMachine":
            self.listboxE.insert(0,self.None_SVM)
        
        """RandomForestClassifier"""
        def RFC(X_train,y_train):
            RF_model=RandomForestClassifier(n_estimators=100)
            RF_model.fit(X_train, y_train)
            y_pred = RF_model.predict(self.X_test)
            y_train_pred = cross_val_predict(RF_model, X_train, y_train, cv=3) 
            accuracy_score_RFC = accuracy_score(y_train, y_train_pred)
            return accuracy_score_RFC
        self.None_RFC = RFC(self.X_train,self.y_train)
        if tempName=="None" and method == "RandomForestClassifier":
            self.listboxE.insert(0, self.None_RFC)
        """PCA"""
        pca = PCA(n_components=8,svd_solver='full')
        X_reduced = pca.fit_transform(self.X_train)
        X_test_reduced=pca.fit_transform(self.X_train)
        Y_Train=self.y_train
        Y_Test=self.y_test
        """SVM"""
        def SVM_PCA(X_reduced,Y_train):
            PCASVM_Linear_model = Pipeline([
                ("scaler", StandardScaler()),
                ("linear_svc", LinearSVC(C=1000,tol=0.001, loss="hinge", max_iter = 1000000)),
                ])
            PCASVM_Linear_model.fit(X_reduced, Y_Train)
            y_pred=PCASVM_Linear_model.predict(X_reduced)
            accuracy_score_PCASVM = accuracy_score(Y_Train, y_pred)
            return accuracy_score_PCASVM
        self.Svm_PCA = SVM_PCA(X_reduced,Y_Train)
        if tempName=="PCA" and method == "SupportVectorMachine":
            self.listboxE.insert(0,self.Svm_PCA )
        """RandomForestClassifier"""
        def RFC_PCA(X_reduced,Y_Train):
            PCARF_model=RandomForestClassifier(bootstrap= True, criterion= 'gini', max_features= 'auto', n_estimators =50, n_jobs= -1)
            PCARF_model.fit(X_reduced,Y_Train)
            y_pred=PCARF_model.predict(X_reduced)
            accuracy_score_PCARFC = accuracy_score(Y_Train, y_pred)
            return accuracy_score_PCARFC
        self.Rfc_PCA = RFC_PCA(X_reduced,Y_Train)
        if tempName=="PCA" and method == "RandomForestClassifier":
            self.listboxE.insert(0,self.Rfc_PCA)

        """Logictics Regression"""
        def LR_PCA(X_reduced,Y_Train):
            PCALR_model = LogisticRegression(C=0.001, penalty= 'l1', solver= 'saga')
            PCALR_model.fit(X_reduced,Y_Train)
            y_pred=PCALR_model.predict(X_reduced)
            accuracy_score_PCALR = accuracy_score(Y_Train, y_pred)
            return accuracy_score_PCALR
        self.Lr_PCA = LR_PCA(X_reduced,Y_Train)
        if tempName=="PCA" and method == "LogisticRegression":
            self.listboxE.insert(0,self.Lr_PCA)
        """DecisionTree"""
        def DT_PCA(X_reduced,Y_Train):
            PCADT_model = DecisionTreeClassifier(criterion= 'entropy', max_depth= 4, max_features= 'auto', min_samples_leaf= 2)
            PCADT_model = PCADT_model.fit(X_reduced,Y_Train)
            y_pred=PCADT_model.predict(X_reduced)
            accuracy_score_PCADT = accuracy_score(Y_Train, y_pred)
            return accuracy_score_PCADT
        self.Dt_PCA = DT_PCA(X_reduced,Y_Train)
        if tempName=="PCA" and method == "DecisionTreeClassifier":
            self.listboxE.insert(0,self.Dt_PCA)
        """OHD"""
        data = pd.read_csv("income.csv")
        OHEdata = pd.get_dummies(data)
        data=data.drop(['educational-num'], axis = 1)
        total = data.isnull().sum()
        percent = (data.isnull().sum()/data.isnull().count()*100)
        missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        missing_data.head(15)
        data = data.fillna(value={'workclass':'Private','occupation':'Craft-repair','native-country':'United-States'}) 
        OHEdata = pd.get_dummies(data)
        std = StandardScaler()
        data_std = std.fit_transform(OHEdata)
        X = OHEdata.drop(['income_>50K'], axis = 1)
        y = OHEdata['income_>50K']
        pca = PCA(n_components=40,svd_solver='full')
        data_pca = pca.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(data_pca, y, test_size = 0.3, random_state = 101)
        """Logictics Regression"""
        def OHD_LR(X_train,y_train):
            LR_OHE_model = LogisticRegression(C = 10000, solver='liblinear', random_state=0, max_iter=1000)
            LR_OHE_model.fit(X_train, y_train)
            y_pred = LR_OHE_model.predict(X_train) #Kết quả dự đoán tập test
            accuracy_score_LROHD = accuracy_score(y_train, y_pred)
            return accuracy_score_LROHD
        self.Ohd_LR = OHD_LR(X_train,y_train)
        if tempName=="OHD" and method == "LogisticRegression":
            self.listboxE.insert(0,self.Ohd_LR)
        """DecisionTree"""
        def OHD_DT(X_train,y_train):
            DT_OHE_model = DecisionTreeClassifier(criterion= 'entropy', max_depth= 4, max_features= 'auto', min_samples_leaf= 2)
            DT_OHE_model = DT_OHE_model.fit(X_train, y_train)
            y_pred = DT_OHE_model.predict(X_train)
            accuracy_score_DTOHD = accuracy_score(y_train, y_pred)
            return accuracy_score_DTOHD
        self.Ohd_DT = OHD_DT(X_reduced,Y_Train)
        if tempName=="OHD" and method == "DecisionTreeClassifier":
            self.listboxE.insert(0,self.Ohd_DT)
        """RandomForestClassifier"""
        def OHD_RF(X_train,y_train):
            RF__OHE_model=RandomForestClassifier(bootstrap= True, criterion= 'gini', max_features= 'auto', n_estimators =50, n_jobs= -1)
            RF__OHE_model.fit(X_train, y_train)
            y_pred = RF__OHE_model.predict(X_train)
            accuracy_score_RFOHD = accuracy_score(y_train, y_pred)
            return accuracy_score_RFOHD
        self.Ohd_RF = OHD_RF(X_reduced,Y_Train)
        if tempName=="OHD" and method == "RandomForestClassifier":
            self.listboxE.insert(0,self.Ohd_RF)
        """SVM"""
        def OHD_SVM(X_train,y_train):
            SVM_Linear_OHE_model = Pipeline([
            ("scaler", StandardScaler()),
            ("linear_svc", LinearSVC()),
            ])
            SVM_Linear_OHE_model.fit(X_train, y_train)
            y_pred = SVM_Linear_OHE_model.predict(X_train)
            accuracy_score_SVM_LinearOHD = accuracy_score(y_train, y_pred)
            return accuracy_score_SVM_LinearOHD
        self.Ohd_SVM = OHD_SVM(X_reduced,Y_Train)
        if tempName=="OHD" and method == "SupportVectorMachine":
            self.listboxE.insert(0,self.Ohd_SVM )
        
    def chart(self):
        # divisions = ["LR","DT","SVM","RF"]
        # divisions_PCA = [self.Lr_PCA,self.Dt_PCA,self.Svm_PCA,self.Rfc_PCA]
        # divisions_OHD = [self.Ohd_LR,self.Ohd_DT,self.Ohd_SVM,self.Ohd_RF]
        # divisions_None = [self.None_LR,self.None_DT,self.None_SVM,self.None_RFC]
        # index = np.arange(4)
        # width = 0.3
        # plt.bar(index,divisions_PCA,width,color = "red",label="PCA") 
        # plt.bar(index+width,divisions_OHD,width,color = "green",label="OHD") 
        # plt.bar(index+2*width,divisions_None,width,color = "black",label="None")
        # plt.title("Bảng so sánh kết quả")
        # plt.ylabel("ACC")
        # plt.xlabel("Method")
        # plt.xticks(index+width/2,divisions)
        # plt.legend(loc="best")
        # plt.show()
        divisions = ["LR","DT","SVM","RF"]
        divisions_PCA = [0.7869,0.8125,0.7965,0.9995]
        divisions_OHD = [0.8509,0.8076,0.8506,0.9994]
        divisions_None = [0.8029,0.8095,0.8032,0.8569]
        index = np.arange(4)
        width = 0.2
        plt.bar(index,divisions_PCA,width,color = "red",label="PCA") 
        plt.bar(index+width,divisions_OHD,width,color = "green",label="OHD") 
        plt.bar(index+2*width,divisions_None,width,color = "black",label="None")
        plt.title("Bảng so sánh kết quả")
        plt.ylabel("ACC")
        plt.xlabel("Method")
        plt.xticks(index+width/2,divisions)
        plt.legend(loc="best")
        plt.show()
GUI(read_data)
