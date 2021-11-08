 clc 
 clear all

M=load('iris_log.dat');
M = M(randperm(size(M,1)),:);%embaralha as linhas
Treino=M(1:105,:);
Teste=M(106:end,:);      %matriz para teste(30%)

net=feedforwardnet(10); %cria a rede e dita o numero de camadas
net=train(net,Treino(:,1:4)',Treino(:,5:7)'); %treina a rede
view(net)
y = net(Teste(:,1:4)') %matriz de teste(resultados
