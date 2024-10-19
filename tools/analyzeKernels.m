close all;
clear all;
addpath('../data/');
kernel1stOrder= table2array(readtable('kernel1stOrder.csv'));
kernel2ndOrder= table2array(readtable('kernel2ndOrder.csv'));
kernel2ndOrder_4= table2array(readtable('kernel2ndOrder_4.csv'));
kernel4thOrder= table2array(readtable('kernel4thdOrder.csv'));

PDDOKernelMesh1stOrder = table2array(readtable('PDDOKernelMesh1stOrder.csv'));
PDDOKernelMesh2ndOrder= table2array(readtable('PDDOKernelMesh2ndOrder.csv'));
PDDOKernelMesh2ndOrder_4= table2array(readtable('PDDOKernelMesh2ndOrder_4.csv'));
PDDOKernelMesh4thOrder= table2array(readtable('PDDOKernelMesh4thdOrder.csv'));

figure; plot(PDDOKernelMesh1stOrder(:,1), PDDOKernelMesh1stOrder(:,2) ,'o'); 
grid on;
axis padded
x = [0.3 0.5];
y = [0.6 0.52];
annotation('textarrow',x,y,'String',' PUT ','FontSize',13,'Linewidth',2, 'Color','r')
title('Example of Kernel')



figure; surf(kernel1stOrder);title('First Order Kernel')
figure; surf(kernel2ndOrder);title('Second Order')
figure; surf(kernel2ndOrder_4);title('Second Order 4') 
figure; surf(kernel4thOrder)