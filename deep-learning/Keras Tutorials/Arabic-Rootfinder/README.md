# Arabic Rootfinder

A fun little project to play with Jupyter Notebooks, Scikit-learn, and neural nets with Keras.

#### Goal

To train a neural network to learn Arabic morphology.

#### Includes:

* Scripts for data mining
* Starter data
* 2 iterations of the model: roots.ipynb and roots-with-noroots.ipynb.

`roots-with-noroots.ipynb` is named weird, but it just means we are more intelligent about tracking words that are "mabniyy", or undeclined.

It's not very accurate (about 50%) so it's pretty addictive to work on. Surely, someone, somewhere, has done this better, but we aren't solving world hunger here, just having some nerdy fun.

Pull requests welcome :)

#### Sample output

The output is formatted as "accuracy: [correctAnswer, input]" with output on the end if incorrect.

```
Correct: ['تبديلي', 'بدل']
Missed:  ['الألياف', 'ليف'] Predicted: للف
Correct: ['تنْويت', 'نوت']
...
Correct: ['متوسّط', 'وسط']
Correct: ['تنفيذي', 'نفذ']
Missed:  ['المتبقية', 'بقا'] Predicted: ربب
Missed:  ['متقدم', 'قدم'] Predicted: ودم
Missed:  ['الأساسي', ''] Predicted: سسس
Correct: ['تغيير', 'غير']
Correct: ['سطر', 'سطر']
Correct: ['سليم', 'سلم']
Correct: ['جملة', 'جمل']
Missed:  ['الذهب', 'ذهب'] Predicted: هبب
Correct: ['طرفية', 'طرف']
Correct: ['متوسطة', 'وسط']
Missed:  ['سوق', 'سوق'] Predicted: وقق
Score: 58.8%
```

#### How to use

Just install Jupyter Notebook and run `jupyter notebook` in this folder, and select one of the `ipynb` files.
