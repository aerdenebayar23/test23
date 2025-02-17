# -*- coding: utf-8 -*-
"""deeplearning_lab1convolving.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1w5CufelZ1flUpA0o5IX7FFH0YLDtZMn0

# Шугаман давхрага

Сэргээн санах CS308. Дараах шугаман давхрагын объектыг гүйцээн бичнэ үү.
"""

from google.colab import drive
drive.mount('/content/drive')

import numpy as np

class Linear_Layer:
    def __init__(self, in_dim, out_dim, alpha=0.01, Theta=None, bias=None):
        self.alpha = alpha
        fac = 1  # Example: setting fac to 1
        if Theta is None:
            self.Theta = np.random.randn(in_dim, out_dim) / fac
        else:
            self.Theta = Theta
        if bias is None:
            self.bias = np.random.randn(out_dim) / fac
        else:
            self.bias = bias
        # Initialize grad and grad_bias to zero
        self.grad = np.zeros_like(self.Theta)
        self.grad_bias = np.zeros_like(self.bias)

    def forward_pass(self, X):
        self.X = X
        self.z = np.dot(X, self.Theta) + self.bias  # Linear transformation
        return self.z

    def backprop(self, grad):
        print(f"ReLU Backprop: self.X.shape = {self.X.shape}, grad.shape = {grad.shape}")
        dTheta = np.dot(self.X.T, grad) / self.X.shape[0]
        dbias = np.sum(grad, axis=0) / self.X.shape[0]

        # Update grad and grad_bias instead of directly updating Theta and bias
        self.grad = dTheta
        self.grad_bias = dbias

        # Calculate and return the gradient for the previous layer
        grad_prev = np.dot(grad, self.Theta.T)
        return grad_prev

    def applying_sgd(self):
        self.Theta = self.Theta - (self.alpha * self.grad)
        self.bias = self.bias - (self.alpha * self.grad_bias)
    def count_parameters(model):
        return sum(p.size for p in model.parameters())

"""# Convolution

Convolutional Neural Networks нь өмнөх хичээлд үзсэн энгийн мэдрэлийн сүлжээнүүдтэй төстэй юм. Эдгээр нь суралцах боломжтой жин, биас бүхий нейронуудаас бүрддэг. Нейрон бүр зарим оролтыг хүлээн авч, цэгийн үржвэр хийж, дурын шугаман бус байдлаар ажилладаг. Сүлжээ нь нэг онооны функцийн уламжлалаар авсан алдаанаас суралцсаар байна. Оролтонд байгаа raw image pixels-ээс нөгөө гаралтанд байгаа ангийн оноо хүртэл урттай байна. Мөн сүүлийн (бүрэн холбогдсон) давхарга дээр (жишээ нь SVM/Softmax) гэх мэт алдагдал функцтэй байгаа бөгөөд тогтмол Neural Networks-ийг сурахад зориулан сүлжээний кодыг бичих нь энэхүү лабораторийн ажилын зорилго.

Тэгвэл ямар өөрчлөлт гардаг вэ? ConvNet архитектурууд нь оролтууд нь зураг гэсэн тодорхой таамаглалыг гаргадаг бөгөөд энэ нь бидэнд тодорхой шинж чанаруудыг архитектурт кодлох боломжийг олгодог. Дараа нь эдгээр нь шууд чиглэлийн функцийг хэрэгжүүлэхэд илүү үр дүнтэй болгож, сүлжээн дэх параметрүүдийн хэмжээг маш ихээр багасгадаг. Тиймээг үүнийг онцлог шинжийг ялгагч гэж нэрлэх нь бас бий. Feature extractor.

CS308 Эргэн санахад: Мэдрэлийн сүлжээ Neural Network нь оролт (нэг вектор) хүлээн авч, хэд хэдэн нуугдмал давхаргуудаар дамжуулан хувиргадаг. Нуугдмал давхарга бүр нь нейронуудын багцаас бүтдэг. Өмнөх давхарга дахь нейрон бүр нь бүх нейронтой бүрэн холбогдсон байдаг үүнийг өөрөөр шугаман давхрага гэж нэрлэдэг. Мөн нэг давхарга дахь нейронууд бүрэн бие даасан байдлаар ажилладаг бөгөөд ямар нэгэн холболтыг хуваалцдаггүй. Хамгийн сүүлд бүрэн холбогдсон давхаргыг "өгөгдлийн давхарга", өөрөөр ангилагч гэж нэрлэдэг бөгөөд ангилалын нөхцөлд энэ нь ангийн оноог илэрхийлдэг.

Тогтмол мэдрэлийн сүлжээ нь бүрэн дүрс том хэмжээний зураг боловсруулахад их хэмжээний параметр, тооцооллын хүндрэл гэх мэт учир дутагдалтай болохоор амьдралд сайн хэмждэггүй. CIFAR-10-т зураг нь зөвхөн 32х32х3 хэмжээтэй (32 өргөн, 32 өндөр, 3 өнгөт суваг) байдаг. Тиймээс мэдрэлийн сүлжээний эхний нуугдмал давхаргад ганц бүрэн холбогдсон нейрон нь 32\*32\*3 = 3072 оролт нь жинтэй байна. Энэ хэмжээг зохицуулах боломжтой мэт боловч бүрэн холбогдсон энэ сүлжээ нь илүү том дүрсүүдэд одоогын технологийн шийдэл боломжгүй болох нь тодорхой. Жишээ нь: илүү том хэмжээтэй зураг, 200x200x3, 200\*200\*3 = 120,000 жинтэй нейронууд руу хөтөлнө. Түүнээс гадна, бид хэд хэдэн ийм нейронтой давхрагатай болсоноор сайн загвар олж авах нь гарцаагүй. Тиймээс параметрүүд экспонциалаар нэмэгдэх болно!

3D хэмжээний нейронууд. Convolutional Neural Network-ийн оролт нь дүрсүүдээс бүрдэж, архитектурыг илүү ухаалаг аргаар хязгаарладаг гэдгээрээ давуу талтай. Ялангуяа тогтмол нейроны сүлжээнүүдийн ялгаатай нь ConvNet-ийн давхарга нь өргөн, өндөр, гүн гэсэн 3 хэмжээст нейронуудыг зохион байгуулсан байдаг. (Энд гүн гэдэг үг нь сүлжээнд нийт давхаргын тоог хэлж байна) Жишээ нь: CIFAR-10 дахь оролтын зураг нь идэвхижүүлэлтийн оролтын хэмжээ бөгөөд багтаамж нь 32х32х3 (өргөн, өндөр, гүн) хэмжээстэй байна. Бидний удахгүй харах болно. Давхарга дахь нейронууд нь бүх нейронуудыг бүрэн холбох маягаар биш харин түүнээс өмнө зөвхөн давхаргын жижиг хэсэгтэй холбогдоно. Түүнээс гадна, CIFAR-10-ийн эцсийн гаралтын давхарга нь 1x1x10 хэмжээтэй байх болно. Учир нь ConvNet архитектурын төгсгөлөөр бид бүрэн зургийг гүнзгий хэмжээсээр зохион байгуулсан нэг вектор болгон багасгах болно.

![Alt Text](https://cs231n.github.io/assets/nn1/neural_net2.jpeg)

![Alt Text](https://cs231n.github.io/assets/cnn/cnn.jpeg)

Бидний дээр өгүүлсэнчлэн энгийн ConvNet нь давхаргын дараалал бөгөөд ConvNet-ийн давхарга бүр нь ялгаатай функцээр дамжуулан нэг хэмжээний идэвхжүүлэлтийг нөгөөд хувиргадаг. Бид ConvNet архитектурыг бүтээхийн тулд үндсэн гурван төрлийн давхаргыг ашигладаг: Convolutional Layer, Pooling Layer, FC layer (CS308 Neural Network гэх мэт ангилагч давхрагууд). Бид эдгээр давхруудыг ашиглан ConvNet архитектурыг бий болгох болно.

Загвар бүтээх арга барил: Бид доорх дэлгэрэнгүй мэдээллийг авч үзнэ, гэхдээ CIFAR-10 ангилалын энгийн ConvNet нь архитектуртай байж болно [INPUT - CONV - RELU - POOL - FC].

Дэлгэрэнгүй:
- INPUT [32x32x3] нь зургийн түүхий пикселийн утгийг авах бөгөөд энэ тохиолдолд өргөн 32, өндөр 32, гурван өнгөт сувагтай R,G,B гэсэн гурван өнгөний сувагтай дүрсийг өгнө.

- CONV давхарга нь оролт дахь нейронуудын гаралтыг тооцоолно. Тус бүр нь тэдний жин болон оролтын хэмжээнд холбогдсон жижиг бүс нутгийн хооронд цэг үржвэр тооцоолно. Энэ нь 12 шүүлтүүр ашиглахаар шийдсэн бол [32x32x12] зэрэг хэмжээний гаралт гаргана.

- RELU давхарга нь 0-д орох max(0,x) thresholding зэрэг элементийн идэвхижүүлэлтийн функцийг хэрэгжүүлнэ. Энэ нь тухайн оролтын хэмжээг өөрчлөхгүй орхидог ([32х32х12]).

- POOL давхарга нь орон зайн хэмжээс (өргөн, өндөр) дагуу доошлуулах процессыг хэрэгжүүлнэ. Үүний үр дүнд [16x16x12] гэх мэт хэмжээстэй болно.

- FC (өөрөөр хэлбэл бүрэн холбогдсон) давхарга нь хичээлийн оноог тооцоолно. Үүний үр дүнд хэмжээ нь [1x1x10] болно. Энд 10 тоо тус бүр нь CIFAR-10 гэсэн 10 ангилалын дунд зэрэг ангиллын оноотой дүйцэхүйц байна. Жирийн мэдрэлийн сүлжээнүүд болон нэрний утгад байгаа шиг энэ давхарга дахь нейрон бүр өмнөх боть дахь бүх тоонуудтай холбогдоно.

Ийнхүү ConvNets нь анхны зургийн давхаргыг анхны пикселийн утгаасс давхар өөрчилж, эцсийн ангийн оноог авдаг. Зарим давхаргын параметрүүд байдаг бол зарим нь байдаггүй. Ялангуяа, CONV/FC давхарга нь оролтын багтаамжийн идэвхжүүлэлтийн функц төдийгүй параметрүүдийн (нейронуудын жин, биас) функц болох хувирлуудыг гүйцэтгэдэг. Нөгөө талаас RELU/POOL давхарга нь тогтсон функцийг хэрэгжүүлнэ. CONV/FC давхарга дахь параметрүүдийг градиент доош нь сургах болно. Ингэснээр анги нь ConvNet-ийн тооцоолдог оноо нь дүрс бүрт зориулсан сургалтын багцын шошгуудтай нийцнэ.

0-padding ашиглах. Зүүн талд байгаа дээрх жишээнд оролтын хэмжээ 5 байсан ба гаралтын хэмжээ нь тэнцүү байсныг анхаараарай: мөн 5. Энэ нь үр дүнгээ өгсөн. Учир нь бидний хүлээн авах талбай 3 байсан бөгөөд бид 1-ийн нөлөөг ашигласан юм. Хэрэв 0-пад хэрэглэхгүй байсан бол гаралтын хэмжээ нь ердөө 3-ын орон зайн хэмжээтэй байх байсан. Учир нь анхны оролтоор ийм олон нейрон "тохирох" байсан. Ерөнхийдөө 0 padding-ийг P=(F−1)/2 байхаар тохируулна. Энэ үед stride нь S=1 бөгөөд оролтын хэмжээ болон гаралтын хэмжээ нь ижил хэмжээтэй байх болно. Энэ аргаар 0-padding ашиглах нь маш түгээмэл байдаг.

Алхам алхааг хязгаарлана. Орон зайн зохион байгуулалт гиперпараметрүүд харилцан хязгаарлалттай байдгийг дахин анхаараарай. Жишээлбэл оролт нь W=10 хэмжээтэй бол ямар ч zero-padding-ийг ашигладаггүй P=0, мөн шүүлтүүрийн хэмжээ нь F=3, тэгээд stride S=2-ыг ашиглах боломжгүй болно. Учир нь $$(W−F+2P)/S+1=(10−3+0)/2+1=4.5$$ , өөрөөр хэлбэл интеграцч биш юм. Энэ нь нейронууд нь интеграцийн хөндлөн гулд нямбай, тэгш бус "тохирохгүй" гэдгийг илтгэнэ. Тиймээс гиперпараметрийн энэ тохиргоог хүчингүй гэж үздэг бөгөөд ConvNet номын сан нь үүнийг тохируулахын тулд бусад нь онцгой эсвэл 0 хавтанг шидэж болно, эсвэл тохируулахын тулд оролтыг хурааж болно, эсвэл ямар нэгэн зүйл. ConvNet архитектурын хэсгээс харахад, ConvNets-ийг зохих ёсоор томруулна. Ингэснээр бүх хэмжээс "ажиллах" нь жинхэнэ толгой өвдөх болно. Энэ нь 0-padding болон зарим дизайны удирдамжийг ашиглах нь ихээхэн хөнгөвчилнө.

Бодит ертөнцийн жишээ. 2012 онд ImageNet challenge-д түрүүлсэн Krizhevsky et al. архитектур нь хэмжээ дүрсийг хүлээн зөвшөөрсөн [227x227x3]. Эхний Convolutional Layer дээр хүлээн авах талбайн хэмжээ F=11, stride S=4and no zero padding P=0 нейронуудыг ашигласан. (227 - 11)/4 + 1 = 55-аас хойш, Конв давхарга нь К=96-ийн гүнтэй байсан тул Конв давхаргын гаралтын хэмжээ хэмжээ нь [55x5x96]. Энэ ботийн 55\*55\*96 нейрон тус бүр нь оролтын хэмжээнд [11x11x3] хэмжээтэй бүсэд холбогдсон байна. Түүнээс гадна, гүн багана бүрийн 96 нейронууд бүгд оролтын ижил [11x11x3] бүсэд холбогдсон байдаг ч мэдээж өөр өөр жинтэй байдаг. Зугаа цэнгэлийн хувьд, хэрэв та жинхэнэ цаасыг уншвал оролтын дүрсүүд 224х224 байсан гэж мэдэгдэж байна. Энэ нь гарцаагүй буруу юм. Учир нь (224 - 11)/4 + 1 нь интегер биш гэдэг нь тодорхой. Энэ нь ConvNets-ийн түүхэн дэх олон хүнийг будлиулсан бөгөөд юу болсныг бараг мэддэггүй. Миний өөрийн хамгийн сайн таамаглал бол Алекс судалгаан дээр дурдаагүй 3 нэмэлт пикселийн нөлөөг ашигласан явдал юм.


Параметр хуваалцах. Параметр хуваалцах схемийг Convolutional Layers-д параметрүүдийн тоог хянахад ашигладаг. Дээрх бодит ертөнцийн жишээг ашиглан бид анхны Conv Layer-д 55\*55\*96 = 290,400 нейронууд байгаа бөгөөд тус бүр нь 11\*11\*3 = 363 жин, 1 биастай гэж харж байна. Энэ нь хамтдаа зөвхөн ConvNet-ийн эхний давхарга дээр 290400 \* 364 = 105,705,600 параметрүүдийг нэмдэг. Энэ тоо маш өндөр байгаа нь тодорхой.

Бид нэг үндэслэлтэй таамаглал дэвшүүлснээр параметрүүдийн тоог эрс багасгаж чадна гэдэг нь харагдаж байна. Хэрэв нэг онцлог нь ямар нэг орон зайн байрлалаар (x,y) тооцоолоход ашигтай бол өөр байрлалаар (x2,y2) тооцоолох нь бас ашигтай байх ёстой. Өөрөөр хэлбэл, нэг 2 хэмжээст гүнийг гүн хальс гэж (жишээ нь: хэмжээ нь [55х5х96] 96 гүн ширхэгтэй, тус бүр нь [55х55] хэмжээтэй) гэсэн үг юм. Бид гүн ширхэг бүрийн нейронуудыг ижил жин, тал тохой ашиглахыг хязгаарлах гэж байна. Энэ параметр хуваалцах схемээр бидний жишээн дэх анхны Conv Layer одоо ердөө 96 өвөрмөц багц жинтэй байх (нэг нь гүн ширхэг бүрт), нийт 96\*11\*11\*3 = 34,848 өвөрмөц жин буюу 34,944 параметр (+96 biases) байх болно. Үүнээс гадна, бүх 55*55 нейронууд нь одоо нэг параметрүүдийг ашиглах болно. Практикт backpropagation үед чанга яригч бүр градиентыг жиндээ тооцоолно. Харин эдгээр градиентууд нь гүн хэсэг бүрт нэмж, зөвхөн нэг ширхэг жинг шинэчлэх болно.


Хэрэв нэг гүн хэсэг дэх бүх нейронууд ижил жингийн векторыг ашиглаж байгаа бол КОНВ-ийн давхаргын урагшлах дамжуулалтыг оролтын хэмжээ бүхий нейронын жингийн цуваа болгон тооцоолж болно гэдгийг анхаараарай (тиймээс нэр: Convolutional Layer). Тийм ч учраас жин багцуудыг шүүлтүүр (эсвэл кернель) гэж нэрлэх нь түгээмэл байдаг. Энэ нь оролттой нийлэгждэг.

![Alt Text](https://cs231n.github.io/assets/cnn/weights.jpeg)

Крижевскийн et al-ийн сурсан жишээ шүүлтүүрүүд. Энд үзүүлсэн 96 ширхэг шүүлтүүр тус бүр нь хэмжээ нь [11x11x3] бөгөөд тус бүр нь нэг гүн ширхэгт 55\*55 нейронууд хуваалцдаг. Параметрийг хуваалцах таамаглал харьцангуй үндэслэлтэй гэдгийг анхаараарай. Хэрэв зургийн аль нэг газарт хөндлөн ирмэгийг илрүүлэх нь чухал юм бол энэ нь өөр ямар нэг газарт ч мөн мэдрэмтгий байх ёстой. Учир нь дүрсүүдийн орчуулгын инварит бүтэцтэй холбоотойгоор. Тиймээс Conv давхаргын гаралтын хэмжээн дэх 55*55 ялгаатай байрлал бүрт хөндлөн ирмэгийг илрүүлэхийн тулд дахин сурах шаардлагагүй.

![Alt Text](https://cdn-media-1.freecodecamp.org/images/gb08-2i83P5wPzs3SL-vosNb6Iur5kb5ZH43)

Backpropagation. Конволюцийн процессийн (өгөгдөл болон жингийн аль алиных нь хувьд) буцах урсгал нь мөн л конволюц (гэхдээ орон зайгаар эргэдэг шүүлтүүртэй) юм. Энэ нь 1 хэмжээст тохиолдолд 1 хэмжээстийг 1-ийн жишээгээр авахад хялбар байдаг.

1х1-ийн конволюц. Үүнээс гадна хэд хэдэн судалгаанд 1x1 конволюцийг ашигласан. Зарим хүмүүс эхлээд 1x1 конволютыг ялангуяа дохио боловсруулалтаас ирсэн үед харахдаа эргэлзэж тээнэгэлздэг. Ер нь сигнал нь 2 хэмжээст тул 1x1 конволют нь утгагүй (энэ нь зүгээр л цэгийн хэмжүүр). Гэсэн хэдий ч ConvNets-д энэ нь тийм биш юм. Учир нь бид 3 хэмжээст хэмжээнээс илүү ажилладаг гэдгийг санах ёстой бөгөөд шүүлтүүрүүд нь оролтын хэмжээг үргэлж бүрэн гүнзгийгээр сунгадаг. Жишээлбэл, оролт нь [32х32x3] байвал 1x1 конволют хийх нь 3 хэмжээст цэгийн бүтээгдэхүүн (оролтын гүн нь 3 сувагтай тул) үр дүнтэй байх болно.

Dilated convolutions. Саяхны нэгэн ололт (жишээ нь Fisher Yu, Vladlen Koltun судалгааг харна уу) нь dilation хэмээх CONV давхаргад дахиад нэг гиперпараметерийг танилцуулах явдал юм. Одоогоор бид зөвхөн contiguous байгаа CONV шүүлтүүрийн талаар ярилцсан. Гэсэн хэдий ч, dilation гэж нэрлэгдэх эс бүрийн хооронд зайтай шүүлтүүртэй байх боломжтой. Жишээ нь нэг хэмжээст 3 хэмжээтэй шүүлтүүр w нь оролт x-ийн дээр тооцоолно: $$ w[0]*x[0] + w[1]*x[1] + w[2]*x[2] $$ . Энэ бол 0-ийн дилаци юм. 1-р дилацийн хувьд шүүлтүүр нь оронд нь $$ w[0]*x[0] + w[1]*x[2] + w[2]*x[4] $$ -ийг тооцоолно; Өөрөөр хэлбэл хэрэглээний хооронд 1-ийн ялгаа байдаг. Энэ нь 0-дипляцтай шүүлтүүртэй хослуулан ашиглах зарим тохиргоонуудад маш их ашиг тустай байж болно. Учир нь энэ нь оролтуудаар орон зайн мэдээллийг илүү хатуугаар цөөн давхаргатай нэгтгэх боломжийг олгодог. Жишээ нь, хэрэв та хоёр 3x3 CONV давхаргыг бие биенийхээ дээр давхарлах юм бол 2-р давхарга дээрх нейронууд нь оролтын 5х5 патчын функц гэдгийг өөртөө итгүүлж болно (бид эдгээр нейронуудын үр дүнтэй хүлээн авах талбар нь 5x5 гэж хэлэх болно). Хэрэв бид dilation конволюцийг ашиглавал энэ үр дүнтэй хүлээн авах талбар илүү хурдан сурах болно.
"""

import numpy as np

class ConvLayer:
    def __init__(self, num_filters=8, filter_dim=3, stride=1, pad=1, alpha=0.01):
        self.num_filters = num_filters
        self.filter_dim = filter_dim
        self.stride = stride
        self.filter = np.random.randn(self.filter_dim, self.filter_dim)
        self.filter = self.filter / self.filter.sum()  # Normalize filter
        self.bias = np.random.rand() / 10  # Small random bias
        self.pad = pad
        self.alpha = alpha
        self.filters = np.random.randn(num_filters, filter_dim, filter_dim) / np.sqrt(filter_dim * filter_dim)
        self.biases = np.random.randn(num_filters) / 10  # Жижиг санамсаргүй налуу утгууд

    def convolving(self, X, fil):
        """Ганц кернелд convolution хийх"""
        dimen_x = (X.shape[0] - self.filter_dim) // self.stride + 1
        dimen_y = (X.shape[1] - self.filter_dim) // self.stride + 1
        z = np.zeros((dimen_x, dimen_y))

        for i in range(dimen_x):
            for j in range(dimen_y):
                patch = X[i * self.stride:i * self.stride + self.filter_dim,
                          j * self.stride:j * self.stride + self.filter_dim]
                z[i, j] = np.sum(patch * fil)

        return z
    def forward_pass(self, X):
        """Олон кернелээр convolution хийх"""
        self.X = X
        batch_size, height, width = X.shape
        output_height = (height - self.filter_dim) // self.stride + 1
        output_width = (width - self.filter_dim) // self.stride + 1
        self.z = np.zeros((batch_size, self.num_filters, output_height, output_width))

        for b in range(batch_size):
            for f in range(self.num_filters):
                self.z[b, f] = self.convolving(self.X[b], self.filters[f]) + self.biases[f]

        return self.z

    def backprop(self, grad_z):
        """Олон кернелд зориулж backprop хийх"""
        batch_size, num_filters, grad_h, grad_w = grad_z.shape
        self.grads = np.zeros_like(self.X)  # Оролт руу буцаах градиент
        self.grad_filters = np.zeros_like(self.filters)
        self.grad_biases = np.zeros_like(self.biases)

        for b in range(batch_size):
            for f in range(self.num_filters):
                # Гаралтын градиентийг жинтэй convolution хийх
                filter_flipped = np.flip(self.filters[f], axis=(0, 1))
                self.grads[b] += self.convolving(np.pad(grad_z[b, f], ((1, 1), (1, 1)), 'constant'), filter_flipped)

                # Кернелүүдийн градиент тооцоолох
                for i in range(self.filter_dim):
                    for j in range(self.filter_dim):
                        self.grad_filters[f, i, j] += np.sum(
                            grad_z[:, f, :, :] * self.X[:, i:i+grad_h, j:j+grad_w]
                        )

                # Налууны градиент
                self.grad_biases[f] += np.sum(grad_z[:, f, :, :])

        return self.grads

    def applying_sgd(self):
        # Update filter and bias using SGD
        self.filter = self.filter - (self.alpha * self.grad_filter)
        self.bias = self.bias - (self.alpha * self.grad_bias)

    def change_alpha(self):
        # Reduce learning rate by a factor of 10
        self.alpha = self.alpha / 10

"""## Даалгавар

1. Кодыг гүйцээн ажиллуулах
2. Олон кернэл дээр ажилладаг болгох

## Pooling давхрага
![Alt Text](https://miro.medium.com/v2/resize:fit:720/format:webp/1*fXxDBsJ96FKEtMOa9vNgjA.gif)


ConvNet архитектурт дараалсан Conv давхаргуудын хооронд pooling давхаргыг үе үе оруулах нь түгээмэл байдаг. Түүний функц нь сүлжээн дэх параметр, тооцооллын хэмжээг багасгахын тулд төлөөллийн орон зайн хэмжээг аажмаар багасгах, улмаар тооцооллыг хянах явдал юм. Pooling Layer нь оролтын гүн ширхэг бүр дээр бие даан ажиллаж, MAX-ийн үйл ажиллагааг ашиглан орон зайгаар дахин хийдэг. Хамгийн түгээмэл хэлбэр нь 2х2 хэмжээтэй шүүлтүүртэй Pool давхарга юм. Оролтод орсон гүн хэсэг бүрийг 2-оор нь өргөн, өндрийн аль алинаар нь 2-оор доош буулгаж, идэвхжүүлэлтийн 75%-ийг хаядаг. Энэ тохиолдолд MAX-ийн process бүр нь 4 тооноос дээш (ямар нэг гүнзгий хэсэгт 2x2 бүс) дээд утгыг авах болно. Энд гүн хэмжээс нь өөрчлөгдөөгүй хэвээр байна. Ерөнхийдөө:
-	W1×H1×D1 хэмжээтэй томъёог хүлээн зөвшөөрч байна
-	Хоёр гиперпараметр шаардлагатай:
-	тэдний орон зайн хэмжээ F,
-   S-ийн алхаа,
-	W2×H2×D2 хэмжээтэй хэмжээс гаргадаг. Хаана:
-	W2=(W1−F)/S+1
-	H2=(H1−F)/S+1
-	D2=D1
-	Оролтын тогтмол функцийг тооцоолсноос хойш 0 параметрүүдийг гаргадаг
Pooling layers-ийн хувьд оролтыг zero-padding ашиглан pad хийх нь түгээмэл биш юм. Практикт олдсон max pooling давхаргын хоёр л нийтлэг харагддаг хувьсал байдаг гэдгийг анхаарах нь зүйтэй. Үүнд: F=3,S=2 (мөн давхар усан сан гэж нэрлэдэг), мөн илүү түгээмэл F=2, S=2-той Pool-ийн давхарга. Илүү том хүлээн авах талбайтай Pool хэт хөнөөлтэй.
"""

class Pooling:
    def __init__(self, pool_dim=2, stride=2):
        self.pool_dim = pool_dim
        self.stride = stride

    def forward_pass(self, data):
        (q, p, t) = data.shape
        z_x = int((p - self.pool_dim) / self.stride) + 1
        z_y = int((t - self.pool_dim) / self.stride) + 1
        after_pool = np.zeros((q, z_x, z_y))
        for ii in range(0, q):
            liss = []
            for i in range(0, p, self.stride):
                for j in range(0, t, self.stride):
                    if (i + self.pool_dim <= p) and (j + self.pool_dim <= t):
                        temp = data[ii, i:i + self.pool_dim, j:j + self.pool_dim]
                        temp_1 = np.max(temp)
                        liss.append(temp_1)
            liss = np.asarray(liss)
            liss = liss.reshape((z_x, z_y))
            after_pool[ii] = liss
            del liss
        return after_pool

    def backprop(self, pooled):
        (a, b, c) = pooled.shape
        cheated = np.zeros((a, 2 * b, 2 * c))
        for k in range(0, a):
            pooled_transpose_re = pooled[k].reshape((b * c))
            count = 0
            for i in range(0, 2 * b, self.stride):
                for j in range(0, 2 * c, self.stride):
                    cheated[k, i:i + self.stride, j:j + self.stride] = pooled_transpose_re[count]
                    count = count + 1
        return cheated

    def applying_sgd(self):
        pass

"""## Softmax

Шууд чиглэл


![Alt Text](https://e2eml.school/images/softmax/def_01_eq.png)


Буцах чиглэл

![Alt Text](https://e2eml.school/images/softmax/d_softmax_03.png)
"""

class softmax:
    def __init__(self):
        pass
    def expansion(self, t):
        (a,) = t.shape
        Y = np.zeros((a,10))
        for i in range(0,a):
            Y[i,t[i]] = 1
        return Y
    def forward_pass(self, z):
        self.z =  z
        (p,t) = self.z.shape
        self.a = np.zeros((p,t))
        for i in range(0,p):
            for ii in range(0,t):
                self.a[i,ii] = None #
        return self.a
    def backprop(self, Y):
        y = self.expansion(Y)
        self.grad = (self.a - y)
        return self.grad
    def applying_sgd(self):
        pass

class relu:
    def __init__(self):
        pass
    def forward_pass(self, z):
        if (len(z.shape) == 3):
            z_temp = z.reshape((z.shape[0], z.shape[1]*z.shape[2]))
            z_temp_1 = self.forward_pass(z_temp)
            self.a_1 = z_temp_1.reshape((z.shape[0], z.shape[1], z.shape[2]))
            return (self.a_1)
        else:
            (p,t) = z.shape
            self.a = np.zeros((p,t))
            for i in range(0,p):
                for ii in range(0,t):
                        self.a[i,ii] = max([0,z[i,ii]])
            return self.a
    def derivative(self, a):
        if a>0:
            return 1
        else:
            return 0
    def backprop(self, grad_previous):
        if (len(grad_previous.shape)==3):
            (d, p, t) = grad_previous.shape
            self.grad = np.zeros((d, p, t))
            for i in range(d):
                for ii in range(p):
                    for iii in range(t):
                        self.grad[i, ii, iii] = (grad_previous[i, ii, iii] * self.derivative(self.a_1[i, ii, iii]))
            return (self.grad)
        else:
            (p,t) = grad_previous.shape
            self.grad = np.zeros((p,t))
            for i in range(p):
                for ii in range(t):
                    self.grad[i,ii] = grad_previous[i,ii] * self.derivative(self.a[i,ii])
            return (self.grad)
    def applying_sgd(self):
        pass

class padding():
    def __init__(self, pad = 1):
        self.pad = pad
    def forward_pass(self, data):
        X = np.pad(data , ((0, 0), (self.pad, self.pad), (self.pad, self.pad)),'constant', constant_values=0)
        return X
    def backprop(self, y):
        return (y[:, 1:(y.shape[1]-1),1:(y.shape[2]-1)])
    def applying_sgd(self):
        pass

class reshaping:
    def __init__(self):
        pass
    def forward_pass(self, a):
        self.shape_a = a.shape
        self.final_a = a.reshape(self.shape_a[0], self.shape_a[1]*self.shape_a[2])
        return self.final_a
    def backprop(self, q):
        return (q.reshape(self.shape_a[0], self.shape_a[1], self.shape_a[2]))
    def applying_sgd(self):
        pass

class cross_entropy:
    def __init__(self):
        pass
    def expansion(self, t):
        (a,) = t.shape
        Y = np.zeros((a,10))
        for i in range(0,a):
            Y[i,t[i]] = 1
        return Y
    def loss(self, A, Y):
        exp_Y = self.expansion(Y)
        (u,i) = A.shape
        loss_matrix = np.zeros((u,i))
        for j in range(u):
            for jj in range(i):
                if exp_Y[j,jj] == 0:
                    loss_matrix[j,jj] = np.log(1 - A[j,jj])
                else:
                    loss_matrix[j,jj] = np.log(A[j,jj])
        return ((-(loss_matrix.sum()))/u)

class accuracy:
    def __init__(self):
        pass
    def value(self, out, Y):
        self.out = np.argmax(out, axis=1)
        p = self.out.shape[0]
        total = 0
        for i in range(p):
            if Y[i]==self.out[i]:
                total += 1
        return total/p

class ConvNet:
    def __init__(self, Network):
        self.Network = Network
    def forward_pass(self, X):
        n = X
        for i in self.Network:
            n = i.forward_pass(n)
            # print(n.shape) #
        return n
    def backprop(self, Y):
        m = Y
        count = 1
        for i in (reversed(self.Network)):
            m = i.backprop(m)
    def applying_sgd(self):
        for i in self.Network:
            i.applying_sgd()

from tensorflow.keras.datasets import mnist

(Xtr, Ytr), (Xte, Yte) = mnist.load_data()
X_training = Xtr[:1000,:,:]
Y_training = Ytr[:1000]
X_training = X_training/255
al = 0.3
stopper = 85.0

complete_NN = ConvNet([
                        padding(),
                        ConvLayer(),
                        Pooling(),
                        relu(),
                        padding(),
                        ConvLayer(),
                        Pooling(),
                        relu(),
                        ConvLayer(),
                        relu(),
                        reshaping(),
                        Linear_Layer(5*5, 24, alpha = al),
                        relu(),
                        Linear_Layer(24, 10, alpha = al),
                        softmax()])

CE = cross_entropy()
acc = accuracy()
epochs = 3
broke = 0
batches = 100
for i in range(epochs):
    k = 0
    for j in range(batches, 1001, batches):
        out = complete_NN.forward_pass(X_training[k:j])
        print("epoch:{} \t batch: {} \t loss: \t {}".format(i+1, int(j/batches), CE.loss(out, Y_training[k:j])), end="\t")
        accur = acc.value(out, Y_training[k:j])*100
        print("accuracy: {}".format(accur))
        if accur >= stopper:
            broke = 1
            break
        complete_NN.backprop(Y_training[k:j])
        complete_NN.applying_sgd()
        k = j
    if broke == 1:
        break

out = complete_NN.forward_pass(X_training)
print("The final loss is {}".format(CE.loss(out, Y_training)))
print("The final accuracy on train set is {}".format(acc.value(out, Y_training)*100))

Xtest = Xte/255
out_1 = complete_NN.forward_pass(Xtest)
print("The accuracy on test set is {}".format(acc.value(out_1, Yte)*100))

"""# Даалгавар

3. Загварын суралцаж буй параметрүүдийн тоог гаргах
4. Сургалтын өгөгдлийг бүгдийг ашиглан сургалт хийх
5. Илүү гүн загварт сургалт хийх
6. Үр дүнг илүү сайн гаргах талаарх дүгнэлт хийх
"""