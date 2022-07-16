clear;
close all;
clc;
%% normal example of estimation
% parameters for x

N = 128;
n = 0:1:N-1;
A = 1;
f0 = 0.1917;         
phi = (pi/7);
x = A*cos(2*pi*f0*n + phi);% equuation

SNRdb = 30; %Given SNRdb
SNRdb10 = SNRdb/10;
SNR = 10^(SNRdb10);
variance = (A^2)/(2*SNR);%from given relation
W = sqrt(variance).*randn(size(x)); %Gaussian white noise W
x = x + W; %Add the noise


% finding coarse frequency


L = length(x);
N = L;
xd = x(1,1:L);
wn = kaiser(L,5);
wn = wn';%kaiser window decreases the spectral energy distribution
xwn = xd.*wn;
xwn = [xwn,zeros(1,(N-L))];
Xk = fft(xwn);

% finding frequency from dft

[peaks,locs]=findpeaks(abs(Xk));
[peaks_sorted,locs_sorted] = sort(peaks,'descend');
plocs = locs - ones(size(locs));
f0c = (plocs(locs_sorted(1))/N);



% finding xmn and xfn for K;


n = 0:1:(length(x)-1);
expo = exp((1.i)*(2*pi)*(f0c).*n); 
xmn = x.*expo;
Xmk = fft(xmn);


% took K=0 to remove negative frequency 
% high pass filter
% given in paper to pass this for removing negative frequencies
K = 0;
Xmk(1,1:K+1) =0;
Xmk(1,128:(127-K))=0;

% reconstruction of xfn with positive frequency only

xfne = ifft(Xmk);
xfn = xfne.*(1./expo);
xfk = fft(xfn);


% Fine frequency estimation 

f1 = frequencyalgo1(xfn,1000);
f2 = frequencyalgo2(xfn,1000);
fprintf("THE FREQUENCY ESTIMATION FROM EXISTING METHOD (or) COARSE FREQUENCY = %.4f \n",f0c);
fprintf("THE FREQUENCY ESTIMATION BY ALGORITHM ONE IS = %.4f \n",f1);
fprintf("THE FREQUENCY ESTIMATION BY ALGORITHM TWO IS = %.4f\n",f2);
%% change in SNR
averageerror = zeros(2,31);
f1 =  zeros(1,100);
f2 = zeros(1,100);
for j = -10:2:50
     for i = 1:1:100
% parameters for x

N = 128;
n = 0:1:N-1;
A = 1;
f0 = 0.1917;         
phi = (pi/7);

 
  
  
    
         
         SNRdb = j; %Given SNRdb
        x = A*cos(2*pi*f0*n + phi);% equuation
        SNRdb10 = SNRdb/10;
        SNR = 10^(SNRdb10);
        variance = (A^2)/(2*SNR);%from given relation
        W = sqrt(variance).*randn(size(x)); %Gaussian white noise W
        x = x + W; %Add the noise


        % finding coarse frequency


        L = length(x);
        N = L;
        xd = x(1,1:L);
        wn = kaiser(L,5);
        wn = wn';%kaiser window decreases the spectral energy distribution
        xwn = xd.*wn;
        xwn = [xwn,zeros(1,(N-L))];
        Xk = fft(xwn);

        % finding frequency from dft

        [peaks,locs]=findpeaks(abs(Xk));
        [peaks_sorted,locs_sorted] = sort(peaks,'descend');
        plocs = locs - ones(size(locs));
        f0c = (plocs(locs_sorted(1))/N);



        % finding xmn and xfn for K;


        n = 0:1:(length(x)-1);
        expo = exp((1.i)*(2*pi)*(f0c).*n); 
        xmn = x.*expo;
        Xmk = fft(xmn);


        % took K=0 to remove negative frequency 
        % high pass filter
        % given in paper to pass this for removing negative frequencies
        K = 0;
        Xmk(1,1:K+1) =0;
        Xmk(1,128:(127-K))=0;

        % reconstruction of xfn with positive frequency only

        xfne = ifft(Xmk);
        xfn = xfne.*(1./expo);
        xfk = fft(xfn);


        % Fine frequency estimation here we are finding absolute errors 

        f1(1,i) = abs(frequencyalgo1(xfn,100)-0.1917);
        f2(1,i) = abs(frequencyalgo2(xfn,100)-0.1917);
        
     end
        averageerror(1,((12+j))/2) = mean(f1);
        averageerror(2,((12+j))/2) = mean(f2);
end      
figure(1);
S = -10:2:50;
R1 = 1.*averageerror(1,:);
R2 = 1.*averageerror(2,:);
stem(S,R1);
hold on;
stem(S,R2);
xlabel("SNR(db)");
ylabel("AVERAGE ERROR IN FREQUENCY ESTIMATION");
title("phase = pi/7,f0=0.1917,N = 128 CHANGE IN SNR ");
legend("ALGO1","ALGO2");

%% CHANGE IN N


% parameters for x
averageerror = zeros(2,31);
f1 = zeros(1,100);
f2 = zeros(1,100);
 phi = pi/7;
 
 A = 1;
 f0 = 0.1917;

for i = 50:15:500
    
     for j = 1:1:100
       N = i;
        n = 0:1:N-1;
        x = A*cos(2*pi*f0*n + phi);% equuation
         
        SNRdb = 30; %Given SNRdb
        SNRdb10 = SNRdb/10;
        SNR = 10^(SNRdb10);
        variance = (A^2)/(2*SNR);%from given relation
        W = sqrt(variance).*randn(size(x)); %Gaussian white noise W
        x = x + W; %Add the noise


        % finding coarse frequency


        L = length(x);
        N = L;
        xd = x(1,1:L);
        wn = kaiser(L,5);
        wn = wn';%kaiser window decreases the spectral energy distribution
        xwn = xd.*wn;
        xwn = [xwn,zeros(1,(N-L))];
        Xk = fft(xwn);
         
        % finding frequency from dft

        [peaks,locs]=findpeaks(abs(Xk));
        [peaks_sorted,locs_sorted] = sort(peaks,'descend');
        plocs = locs - ones(size(locs));
        f0c = (plocs(locs_sorted(1))/N);
        


        % finding xmn and xfn for K;


        n = 0:1:(length(x)-1);
        expo = exp((1.i)*(2*pi)*(f0c).*n); 
        xmn = x.*expo;
        Xmk = fft(xmn);
         

        % took K=0 to remove negative frequency 
        % high pass filter
        % given in paper to pass this for removing negative frequencies
        K = 0;
        Xmk(1,1:K+1) =0;
        Xmk(1,128:(127-K))=0;

        % reconstruction of xfn with positive frequency only

        xfne = ifft(Xmk);
        xfn = xfne.*(1./expo);
        xfk = fft(xfn);
        

        % absolute errors from frequency estimation

         f1(1,j) = abs(frequencyalgo1(xfn,100)-0.1917);
         f2(1,j) = abs(frequencyalgo2(xfn,100)-0.1917);
         
         
     end
         averageerror(1,((i-35)/15)) = mean(f1);
         averageerror(2,((i-35)/15)) = mean(f2);
end
figure(2);
subplot(2,1,1);
S = 50:15:500;
R1 = 1.*averageerror(1,:);
R2 = 1.*averageerror(2,:);
stem(S,R1);
hold on;
stem(S,R2);
xlabel("N------>");
ylabel("AVERAGE ABSOLUTE ERROR IN FREQUENCY ESTIMATION");
title("phase = pi/7,f0=0.1917,SNR = 30,CHANGE IN N");
legend("ALGO1","ALGO2");

%% CHANGE IN N

% parameters for x
averageerror = zeros(2,31);
f1 = zeros(1,100);
f2 = zeros(1,100);
 phi = pi/7;
 
 A = 1;
 f0 = 0.1917;

for i = 50:15:500
    
     for j = 1:1:100
       N = i;
        n = 0:1:N-1;
        x = A*cos(2*pi*f0*n + phi);% equuation
         
        SNRdb = 10; %Given SNRdb
        SNRdb10 = SNRdb/10;
        SNR = 10^(SNRdb10);
        variance = (A^2)/(2*SNR);%from given relation
        W = sqrt(variance).*randn(size(x)); %Gaussian white noise W
        x = x + W; %Add the noise


        % finding coarse frequency


        L = length(x);
        N = L;
        xd = x(1,1:L);
        wn = kaiser(L,5);
        wn = wn';%kaiser window decreases the spectral energy distribution
        xwn = xd.*wn;
        xwn = [xwn,zeros(1,(N-L))];
        Xk = fft(xwn);
         
        % finding frequency from dft

        [peaks,locs]=findpeaks(abs(Xk));
        [peaks_sorted,locs_sorted] = sort(peaks,'descend');
        plocs = locs - ones(size(locs));
        f0c = (plocs(locs_sorted(1))/N);
        


        % finding xmn and xfn for K;


        n = 0:1:(length(x)-1);
        expo = exp((1.i)*(2*pi)*(f0c).*n); 
        xmn = x.*expo;
        Xmk = fft(xmn);
         

        % took K=0 to remove negative frequency 
        % high pass filter
        % given in paper to pass this for removing negative frequencies
        K = 0;
        Xmk(1,1:K+1) =0;
        Xmk(1,128:(127-K))=0;

        % reconstruction of xfn with positive frequency only

        xfne = ifft(Xmk);
        xfn = xfne.*(1./expo);
        xfk = fft(xfn);
        

        % Fine frequency estimation and finding the absolute errors
         f1(1,j) = abs(frequencyalgo1(xfn,100)-0.1917);
         f2(1,j) = abs(frequencyalgo2(xfn,100)-0.1917);
         
         
     end
         averageerror(1,((i-35)/15)) = mean(f1);
         averageerror(2,((i-35)/15)) = mean(f2);
end
subplot(2,1,2);
S = 50:15:500;
R1 = 1.*averageerror(1,:);
R2 = 1.*averageerror(2,:);
stem(S,R1);
hold on;
stem(S,R2);
xlabel("N-------->");
ylabel("AVERAGE ABSOLUTE ERROR IN FREQUENCY ESTIMATION");
title("phase = pi/7,f0=0.1917,SNR = 10,CHANGE IN N");
legend("ALGO1","ALGO2");

%% CHANGE IN K


% parameters for x
averageerror = zeros(2,31);
f1 = zeros(1,100);
f2 = zeros(1,100);
 phi = pi/7;
   N = 128;
  n = 0:1:N-1;
 A = 1;
 f0 = 0.1917;

for i = 0:1:30
    
     for j = 1:1:100
     
        x = A*cos(2*pi*f0*n + phi);% equuation
         
        SNRdb = 30; %Given SNRdb
        SNRdb10 = SNRdb/10;
        SNR = 10^(SNRdb10);
        variance = (A^2)/(2*SNR);%from given relation
        W = sqrt(variance).*randn(size(x)); %Gaussian white noise W
        x = x + W; %Add the noise


        % finding coarse frequency


        L = length(x);
        N = L;
        xd = x(1,1:L);
        wn = kaiser(L,5);
        wn = wn';%kaiser window decreases the spectral energy distribution
        xwn = xd.*wn;
        xwn = [xwn,zeros(1,(N-L))];
        Xk = fft(xwn);
         
        % finding frequency from dft

        [peaks,locs]=findpeaks(abs(Xk));
        [peaks_sorted,locs_sorted] = sort(peaks,'descend');
        plocs = locs - ones(size(locs));
        f0c = (plocs(locs_sorted(1))/N);
        


        % finding xmn and xfn for K;


        n = 0:1:(length(x)-1);
        expo = exp((1.i)*(2*pi)*(f0c).*n); 
        xmn = x.*expo;
        Xmk = fft(xmn);
         

        % took K=0 to remove negative frequency 
        % high pass filter
        % given in paper to pass this for removing negative frequencies
        K = i;
        Xmk(1,1:K+1) =0;
        Xmk(1,128:(127-K))=0;

        % reconstruction of xfn with positive frequency only

        xfne = ifft(Xmk);
        xfn = xfne.*(1./expo);
        xfk = fft(xfn);
        

        % Fine frequency estimation and finding absolute errors

         f1(1,j) = abs(frequencyalgo1(xfn,100)-0.1917);
         f2(1,j) = abs(frequencyalgo2(xfn,100)-0.1917);
         
         
     end
         averageerror(1,((i+1)/1)) = mean(f1);
         averageerror(2,((i+1)/1)) = mean(f2);
end
figure(3);
subplot(2,1,1);
S = 0:1:30;
R1 = 1.*averageerror(1,:);
R2 = 1.*averageerror(2,:);
stem(S,R1);
hold on;
stem(S,R2);
xlabel("K------>");
ylabel("AVERAGE ABSOLUTE ERROR IN FREQUENCY ESTIMATION");
title("phase = pi/7,f0=0.1917,N = 128,SNR =30,CHANGE IN K");
legend("ALGO1","ALGO2");
%%

% parameters for x
averageerror = zeros(2,31);
f1 = zeros(1,100);
f2 = zeros(1,100);
 phi = pi/7;
   N = 128;
  n = 0:1:N-1;
 A = 1;
 f0 = 0.1917;

for i = 0:1:30
    
     for j = 1:1:100
     
        x = A*cos(2*pi*f0*n + phi);% equuation
         
        SNRdb = 10; %Given SNRdb
        SNRdb10 = SNRdb/10;
        SNR = 10^(SNRdb10);
        variance = (A^2)/(2*SNR);%from given relation
        W = sqrt(variance).*randn(size(x)); %Gaussian white noise W
        x = x + W; %Add the noise


        % finding coarse frequency


        L = length(x);
        N = L;
        xd = x(1,1:L);
        wn = kaiser(L,5);
        wn = wn';%kaiser window decreases the spectral energy distribution
        xwn = xd.*wn;
        xwn = [xwn,zeros(1,(N-L))];
        Xk = fft(xwn);
         
        % finding frequency from dft

        [peaks,locs]=findpeaks(abs(Xk));
        [peaks_sorted,locs_sorted] = sort(peaks,'descend');
        plocs = locs - ones(size(locs));
        f0c = (plocs(locs_sorted(1))/N);
        


        % finding xmn and xfn for K;


        n = 0:1:(length(x)-1);
        expo = exp((1.i)*(2*pi)*(f0c).*n); 
        xmn = x.*expo;
        Xmk = fft(xmn);
         

        % took K=0 to remove negative frequency 
        % high pass filter
        % given in paper to pass this for removing negative frequencies
        K = i;
        Xmk(1,1:K+1) =0;
        Xmk(1,128:(127-K))=0;

        % reconstruction of xfn with positive frequency only

        xfne = ifft(Xmk);
        xfn = xfne.*(1./expo);
        xfk = fft(xfn);
        

        % Fine frequency estimation and finding absolute errors

         f1(1,j) = abs(frequencyalgo1(xfn,100)-0.1917);
         f2(1,j) = abs(frequencyalgo2(xfn,100)-0.1917);
         
         
     end
         averageerror(1,((i+1)/1)) = rms(f1);
         averageerror(2,((i+1)/1)) = rms(f2);
end
subplot(2,1,2);
S = 0:1:30;
R1 = 1.*averageerror(1,:);
R2 = 1.*averageerror(2,:);
stem(S,R1);
hold on;
stem(S,R2);
xlabel("K------>");
ylabel("AVERAGE ABSOLUTE ERROR IN FREQUENCY ESTIMATION");
title("phase = pi/7,f0=0.1917,N = 128,SNR =10,CHANGE IN K");
legend("ALGO1","ALGO2");

%%  CHANGE IN PHASE


% parameters for x
averageerror = zeros(2,31);
f1 = zeros(1,100);
f2 = zeros(1,100);

   N = 128;
  n = 0:1:N-1;
 A = 1;
 f0 = 0.1917;

for i = 0:(pi/15):(2*pi)
    
     for j = 1:1:100
         
          phi = i;
        x = A*cos(2*pi*f0*n + phi);% equuation
         
        SNRdb = 30; %Given SNRdb
        SNRdb10 = SNRdb/10;
        SNR = 10^(SNRdb10);
        variance = (A^2)/(2*SNR);%from given relation
        W = sqrt(variance).*randn(size(x)); %Gaussian white noise W
        x = x + W; %Add the noise


        % finding coarse frequency


        L = length(x);
        N = L;
        xd = x(1,1:L);
        wn = kaiser(L,5);
        wn = wn';%kaiser window decreases the spectral energy distribution
        xwn = xd.*wn;
        xwn = [xwn,zeros(1,(N-L))];
        Xk = fft(xwn);
         
        % finding frequency from dft

        [peaks,locs]=findpeaks(abs(Xk));
        [peaks_sorted,locs_sorted] = sort(peaks,'descend');
        plocs = locs - ones(size(locs));
        f0c = (plocs(locs_sorted(1))/N);
        


        % finding xmn and xfn for K;


        n = 0:1:(length(x)-1);
        expo = exp((1.i)*(2*pi)*(f0c).*n); 
        xmn = x.*expo;
        Xmk = fft(xmn);
         

        % took K=0 to remove negative frequency 
        % high pass filter
        % given in paper to pass this for removing negative frequencies
        K = 0;
        Xmk(1,1:K+1) =0;
        Xmk(1,128:(127-K))=0;

        % reconstruction of xfn with positive frequency only

        xfne = ifft(Xmk);
        xfn = xfne.*(1./expo);
        xfk = fft(xfn);
        

        % Fine frequency estimation and finding absolute errors

         f1(1,j) = abs(frequencyalgo1(xfn,100)-0.1917);
         f2(1,j) = abs(frequencyalgo2(xfn,100)-0.1917);
         
         
     end
         p = ((15*i)/pi) + 1;
         p = int16(p);
         averageerror(1,p) = mean(f1);
         averageerror(2,p) = mean(f2);
end
figure(4);
subplot(2,1,1);
S =  0:(pi/15):(2*pi);
R1 = 1.*averageerror(1,:);
R2 = 1.*averageerror(2,:);
stem(S,R1);
hold on;
stem(S,R2);
xlabel("phase------>");
ylabel("AVERAGE ABSOLUTE ERROR IN FREQUENCY ESTIMATION");
title("phase = pi/7,f0=0.1917,N = 128,SNR =30,CHANGE IN K");
legend("ALGO1","ALGO2");

%% CHANGE IN PHASE


% parameters for x
averageerror = zeros(2,31);
f1 = zeros(1,100);
f2 = zeros(1,100);

   N = 128;
  n = 0:1:N-1;
 A = 1;
 f0 = 0.1917;

for i = 0:(pi/15):(2*pi)
    
     for j = 1:1:100
         
          phi = i;
        x = A*cos(2*pi*f0*n + phi);% equuation
         
        SNRdb = 10; %Given SNRdb
        SNRdb10 = SNRdb/10;
        SNR = 10^(SNRdb10);
        variance = (A^2)/(2*SNR);%from given relation
        W = sqrt(variance).*randn(size(x)); %Gaussian white noise W
        x = x + W; %Add the noise


        % finding coarse frequency


        L = length(x);
        N = L;
        xd = x(1,1:L);
        wn = kaiser(L,5);
        wn = wn';%kaiser window decreases the spectral energy distribution
        xwn = xd.*wn;
        xwn = [xwn,zeros(1,(N-L))];
        Xk = fft(xwn);
         
        % finding frequency from dft

        [peaks,locs]=findpeaks(abs(Xk));
        [peaks_sorted,locs_sorted] = sort(peaks,'descend');
        plocs = locs - ones(size(locs));
        f0c = (plocs(locs_sorted(1))/N);
        


        % finding xmn and xfn for K;


        n = 0:1:(length(x)-1);
        expo = exp((1.i)*(2*pi)*(f0c).*n); 
        xmn = x.*expo;
        Xmk = fft(xmn);
         

        % took K=0 to remove negative frequency 
        % high pass filter
        % given in paper to pass this for removing negative frequencies
        K = 0;
        Xmk(1,1:K+1) =0;
        Xmk(1,128:(127-K))=0;

        % reconstruction of xfn with positive frequency only

        xfne = ifft(Xmk);
        xfn = xfne.*(1./expo);
        xfk = fft(xfn);
        

        % Fine frequency estimation and finding the absolute errors

         f1(1,j) = abs(frequencyalgo1(xfn,100)-0.1917);
         f2(1,j) = abs(frequencyalgo2(xfn,100)-0.1917);
         
         
     end
         p = ((15*i)/pi) + 1;
         p = int16(p);
         averageerror(1,p) = mean(f1);
         averageerror(2,p) = mean(f2);
end
subplot(2,1,2);
S =  0:(pi/15):(2*pi);
R1 = 1.*averageerror(1,:);
R2 = 1.*averageerror(2,:);
stem(S,R1);
hold on;
stem(S,R2);
xlabel("phase------>");
ylabel("AVERAGE ABSOLUTE ERROR IN FREQUENCY ESTIMATION");
title("phase = pi/7,f0=0.1917,N = 128,SNR =10,CHANGE IN K");
legend("ALGO1","ALGO2");
%% change in frequency

averageerror = zeros(2,31);
f1 =  zeros(1,100);
f2 = zeros(1,100);
r =1;
for j = 0:0.01:0.3
     for i = 1:1:100
% parameters for x

N = 128;
n = 0:1:N-1;
A = 1;
f0 = j;         
phi = (pi/7);

 %f = zeros(1,31);
  
  
    
         
         SNRdb = 30; %Given SNRdb
        x = A*cos(2*pi*f0*n + phi);% equuation
        SNRdb10 = SNRdb/10;
        SNR = 10^(SNRdb10);
        variance = (A^2)/(2*SNR);%from given relation
        W = sqrt(variance).*randn(size(x)); %Gaussian white noise W
        x = x + W; %Add the noise


        % finding coarse frequency


        L = length(x);
        N = L;
        xd = x(1,1:L);
        wn = kaiser(L,5);
        wn = wn';%kaiser window decreases the spectral energy distribution
        xwn = xd.*wn;
        xwn = [xwn,zeros(1,(N-L))];
        Xk = fft(xwn);

        % finding frequency from dft

        [peaks,locs]=findpeaks(abs(Xk));
        [peaks_sorted,locs_sorted] = sort(peaks,'descend');
        plocs = locs - ones(size(locs));
        f0c = (plocs(locs_sorted(1))/N);



        % finding xmn and xfn for K;


        n = 0:1:(length(x)-1);
        expo = exp((1.i)*(2*pi)*(f0c).*n); 
        xmn = x.*expo;
        Xmk = fft(xmn);


        % took K=0 to remove negative frequency 
        % high pass filter
        % given in paper to pass this for removing negative frequencies
        K = 0;
        Xmk(1,1:K+1) =0;
        Xmk(1,128:(127-K))=0;

        % reconstruction of xfn with positive frequency only

        xfne = ifft(Xmk);
        xfn = xfne.*(1./expo);
        xfk = fft(xfn);


        % Fine frequency estimation here we are finding absolute errors 

        f1(1,i) = abs(frequencyalgo1(xfn,100)-0.1917);
        f2(1,i) = abs(frequencyalgo2(xfn,100)-0.1917);
        
     end
        
        averageerror(1,r) = mean(f1);
        averageerror(2,r) = mean(f2);
        r= r+1;
end      
figure(5);
subplot(2,1,1);
S = 0:0.01:0.3;
R1 = 1.*averageerror(1,:);
R2 = 1.*averageerror(2,:);
stem(S,R1);
hold on;
stem(S,R2);
xlabel("frequency");
ylabel("AVERAGE ERROR IN FREQUENCY ESTIMATION");
title("phase = pi/7,SNR = 30,N = 128 CHANGE IN frequency ");
legend("ALGO1","ALGO2");
%% change in frequency
averageerror = zeros(2,31);
f1 =  zeros(1,100);
f2 = zeros(1,100);
r =1;
for j = 0:0.01:0.3
     for i = 1:1:100
% parameters for x

N = 128;
n = 0:1:N-1;
A = 1;
f0 = j;         
phi = (pi/7);

 %f = zeros(1,31);
  
  
    
         
         SNRdb = 30; %Given SNRdb
        x = A*cos(2*pi*f0*n + phi);% equuation
        SNRdb10 = SNRdb/10;
        SNR = 10^(SNRdb10);
        variance = (A^2)/(2*SNR);%from given relation
        W = sqrt(variance).*randn(size(x)); %Gaussian white noise W
        x = x + W; %Add the noise


        % finding coarse frequency


        L = length(x);
        N = L;
        xd = x(1,1:L);
        wn = kaiser(L,5);
        wn = wn';%kaiser window decreases the spectral energy distribution
        xwn = xd.*wn;
        xwn = [xwn,zeros(1,(N-L))];
        Xk = fft(xwn);

        % finding frequency from dft

        [peaks,locs]=findpeaks(abs(Xk));
        [peaks_sorted,locs_sorted] = sort(peaks,'descend');
        plocs = locs - ones(size(locs));
        f0c = (plocs(locs_sorted(1))/N);



        % finding xmn and xfn for K;


        n = 0:1:(length(x)-1);
        expo = exp((1.i)*(2*pi)*(f0c).*n); 
        xmn = x.*expo;
        Xmk = fft(xmn);


        % took K=0 to remove negative frequency 
        % high pass filter
        % given in paper to pass this for removing negative frequencies
        K = 0;
        Xmk(1,1:K+1) =0;
        Xmk(1,128:(127-K))=0;

        % reconstruction of xfn with positive frequency only

        xfne = ifft(Xmk);
        xfn = xfne.*(1./expo);
        xfk = fft(xfn);


        % Fine frequency estimation here we are finding absolute errors 

        f1(1,i) = abs(frequencyalgo1(xfn,100)-0.1917);
        f2(1,i) = abs(frequencyalgo2(xfn,100)-0.1917);
        
     end
        
        averageerror(1,r) = mean(f1);
        averageerror(2,r) = mean(f2);
        r= r+1;
end      
subplot(2,1,2);
S = 0:0.01:0.3;
R1 = 1.*averageerror(1,:);
R2 = 1.*averageerror(2,:);
stem(S,R1);
hold on;
stem(S,R2);
xlabel("frequency");
ylabel("AVERAGE ERROR IN FREQUENCY ESTIMATION");
title("phase = pi/7,SNR = 30,N = 128 CHANGE IN frequency ");
legend("ALGO1","ALGO2");
