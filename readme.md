# simple CNN
## 설명
`simple-cnn`
* [PyTorch Lecture 10: Basic CNN](https://www.youtube.com/watch?v=LgFNRIFxuUo)을 보고 현재 문법에 맞게 살짝 고쳐서 만든 코드
    * accuracy 98%
`cnn-inception`
* [PyTorch Lecture 11: Advanced CNN](https://www.youtube.com/watch?v=hqYfqNAQIjE&list=PLQ28Nx3M4JrhkqBVIXg-i5_CVVoS1UzAv)을 보고 현재 문법에 맞게 살짝 고쳐서 만든 코드

## Exercise
`simple-cnn`
* 간단한 layer를 하나 늘려서 accuracy를 측정해보았다.
    * full-connected layer에서 320 -> 200, 200 -> 10
        * `Test set: Average loss: 0.0008, Accuracy: 9844/10000 (98%)`
`cnn-inception`
* inception module을 추가로 구현했다. accuracy 반올림해서(ㅋㅋ) 99% 달성
    * `Test set: Average loss: 0.0007, Accuracy: 9854/10000 (99%)`

## 아쉬운점, TBD
* convolutional layer 이후 fully connected layer의 계산
    * dataset의 크기가 몇by몇인지 알수가 없어서..:( 일단 조회하기가 어려웠다 ㅎㅎ..
* 그래프로 learning rate를 나타내기

## 참고
[deep cnn으로 MINST 구현하기](https://wikidocs.net/63618)  
[PyTorch Lecture 10: Basic CNN](https://www.youtube.com/watch?v=LgFNRIFxuUo)  
