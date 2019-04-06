% BE3 - Random signal Processing

% @author:  Thales DELMIRO DE SOUSA
%           Pedro Henrique RODRIGUES DA MOTA

%% INIT
clear all; close all; clc;

%% FIXED PARAMETERS
Tr=1e-3; %(PRI) pulse repetition interval (defined: 1ms)
sigma=1;% Defined :sigma^2=0 dBW

%% VARIABLES


T=Tr/1e1;%pulse duration T<<Tr : user defined
%T<tau<Tr-T
%=>   ktau > kT
kT=1e2;%natural : user defined
Ts=T/kT;%(dt) Time sampling Ts<<T

ktau=5e2;%natural
tau=Ts*ktau;% target round -trip delay
if(ktau<=kT)
    disp('Invalid values ktau should be greater than kT');
end

K=floor(Tr/Ts);%nearest integer towards minus infinity of Tr/ts


%% Question 4: emmited signal 'e'
% Generate and represent the transmitted waveform e as a function of time
% in seconds.

% Defining discrete-time for representation 
t=0:Ts:Tr-Ts; %0:dt:repTime
%generate e
e=rectangularPulse(0,T,t);
% representing e
figure(1); plot(t,e); grid on; box on; xlabel('t[s]'); ylabel('e(t)');
title('Q4: Signal e(t)');


%% Question 5 : SoI
% Generate and represent the SoI vector ?a as a function of time in seconds
% Recall that the amplitude ? is a priori complex valued.

%definig alpha
alpha=4+4*1i;
% defining length of a
a=zeros(1,K);
%generate a(t)=e(t-tau)
a(ktau+1:K)=e(1:K-ktau);
%SoI
SoI=alpha*a;
% representing SoI : Re(alpha*a(t))
figure(2); plot(t,SoI), grid on; box on; xlabel('t[s]'); 
ylabel('Re{ \alpha a(t)}'); title('Q5: SoI Signal Re{ \alpha a(t)}');
% alternativce representation of SoI : Re(alpha*a(t)) x Im(alpha*a(t))
figure(3); plot3(t,real(SoI),imag(SoI)); grid on; box on;
xlabel('Re{ \alpha a(t)}');  ylabel('Im{ \alpha a(t)}'); 
title('Q5: SoI Signal Re( \alpha a(t)) x Im( \alpha a(t)))');


%% Question 6
%6(a) Generate one realization of a real valued Gaussian white noise 'nr'
%normal distriibution parameters

mu=0;
stddev=(sqrt(sigma^2/2));
% Matrix construction
M=zeros(K,1);
C=sigma^2/2*eye(K);
% Generating real valued GWN 'nr'
nr=normrnd(mu,stddev,[K 1]);

%6(b) Generate one realization of a real valued Gaussian white noise 'ni'
% ni=normrnd(M,Var);
ni=normrnd(mu,stddev,[K 1]);

%6(c) Using the results of the preliminary questions of section 1, 
% generate the complex AWGN

% Generating n
n=nr+1i*ni;
% checking var(n)=sigma
vn=var(n);
%6(d)
% Generating x
x=SoI'+n;

% Representating x
figure(4); plot3(t,real(x),imag(x)); grid on; box on; xlabel('t[s]'); 
ylabel('Re( \alpha a(t))'); zlabel('Im( \alpha a(t))');
title('Q6: SoI Signal Re( \alpha a(t))');



%% Question 7
%7(a) Verify that the mean, the variance and the PDF of nr (ni resp.) 
% is conform to the theory
%verifying mean
mnr=mean(nr)
mni=mean(ni)
%verifying var
varnr=var(nr)
varni=var(ni)
%verifying PDF
PDFnr=normpdf(nr,mu,sigma);
figure(5); scatter(nr,PDFnr); grid on; box on; xlabel('nr'); 
ylabel('PDF(nr)');title('Q7: PDF of nr');
PDFni=normpdf(ni,mu,sigma);
figure(6); scatter(ni,PDFni); grid on; box on; xlabel('ni'); 
ylabel('PDF(ni)');title('Q7: PDF of ni');
% verifying histogram
figure(7);histogram(nr); grid on; box on; xlabel('value'); 
ylabel('# realization');title('Q7: Histogram of nr');
figure(8); histogram(ni); grid on; box on; xlabel('value'); 
ylabel('# realization');title('Q7: Histogram of ni');

% 7(b)

fftn=fft(n); % Fast Fourier Transformer 
fs=1/Ts; % reference frequency
Sn=(1/(K*fs)).*(abs(fftn)).^2; %periodogram method 
f=linspace(0,fs,K); % discrete-frequency for representation 
figure(9); plot(f,Sn); grid on; box on; xlabel('f [Hz]'); 
ylabel('Sn [W/Hz]');title('Q7: PSD Sn(f) representation');

%7(c)
%the center of the array as zero-frequency component
Sn_shift=fftshift(fftn);
fshift=(-K/2:K/2-1).*(fs/K); %time shifting
pwrshift=(1/(K*fs)).*abs(Sn_shift).^2; % Sn
figure(10); plot(fshift,pwrshift); grid on; box on; xlabel('f [Hz]'); 
ylabel('Sn[W] centered');title('Q7: PSD Sn(f) representation');
SndB_shift=10*log10(pwrshift); %Sn in dbW
figure(11); plot(fshift,SndB_shift); grid on; box on; xlabel('f [Hz]'); 
ylabel('G(Sn)[dbW] entered');title('Q7: PSD Sn(f) representation');

%% Question 8
%8(a)
% Implementing the matched filter
[Rxe,lag]=xcorr(x,conj(e)); %R(x(t1),e(t2))
%8(b)
% 1)

figure(12); hold on; plot(lag*Ts,Rxe*Ts); 
grid on; box on; xlabel('\tau [s]'); 
ylabel('Rxe'); title('Q8: Matched filter by cross-corrrelation RX/TX');


% 2)
c=3*1e8;%light speed
R=c/2*lag*Ts; %range calculation
[peaks, plocs]=findpeaks(real(Rxe*Ts));
[maxpeak,pos]=max(peaks);

range=R(plocs(pos));

figure(13);hold on;plot(R,Rxe*Ts); scatter(range,maxpeak,'or');
grid on; box on; xlabel( 'Range [m]'); 
leg1=legend('Rxe',...
                    ['Peak of signal: ','Range=',num2str(range),'m'],...
                'Location','northwest' );
            title(leg1,'Legenda')
            legend('boxoff')
ylabel('Rxe'); title('Q8: Matched filter by cross-corrrelation RX/TX');
hold off;

%8(c)
% verifying radar resolution

delta_R=c*T/2


%% Question 12

%% REDEFINING VARIABLES

T=Tr/5;%pulse duration T<<Tr : user defined
%T<tau<Tr-T
%=>   ktau > kT
kT=3e2;%natural : user defined
Ts=T/kT;%(dt) Time sampling Ts<<T

ktau=5e2;%natural
tau=Ts*ktau;% target round -trip delay
if(ktau<=kT)
    disp('Invalid values ktau should be greater than kT');
end

K=floor(Tr/Ts);%nearest integer towards minus infinity of Tr/ts
% Redefining discrete-time for representation 
t=0:Ts:Tr-Ts; %0:dt:repTime

%defining beta
beta=2e8;%B=beta*T
% generating e(t)
e = exp((1i*2*pi*beta/2).*t.^2).*rectangularPulse(0,T,t);
% e=chirp(t,0,1,beta).*rectangularPulse(0,T,t); %alternative waveform generation
% representating e(t)
figure(14); plot3(t,real(e),imag(e)); grid on; box on; xlabel('t [s]'); 
ylabel('Re(e(t))');zlabel('Im(e(t))'); title('Q12: e(t) "chirp" emitted');

%definig alpha
alpha=2+2*1i;
% defining length of a
a=zeros(1,K);
%generate a(t)=e(t-tau)
a(ktau+1:K)=e(1:K-ktau);
%SoI
SoI=alpha*a;
sigma=1;
mu=0;
M=zeros(K);
Var=sigma^2/2*eye(K);
nr=normrnd(mu,stddev,[K 1]);
ni=normrnd(mu,stddev,[K 1]);
%generating n
n=nr+1i*ni;
vn=var(n)
%generating x
x=SoI'+n;
%representating x
figure(15); plot3(t,real(x),imag(x)); grid on; box on; xlabel('t [s]'); 
ylabel('Re(x(t))');zlabel('Im(x(t))'); title('Q12: e(t) "chirp" received');

[Rxe,lag]=xcorr(x,conj(e));
figure(16); plot(lag*Ts,Rxe*Ts); grid on; box on; xlabel('\tau [s]'); 
ylabel('Rxe'); title('Q12: Matched filter by cross-corrrelation RX/TX');

[peaks, plocs]=findpeaks(real(Rxe*Ts));
[maxpeak,pos]=max(peaks);

R=c/2*lag*Ts; %range calculation
range=R(plocs(pos));
figure(17); hold on; plot(R,Rxe*Ts); 
scatter(range,maxpeak,'or');
grid on; box on;
leg1=legend(['Rxe'],...
                    ['Peak of signal: ','Range=',num2str(range),'m'],...
                'Location','northwest' );
            title(leg1,'Legenda')
            legend('boxoff')
xlabel('Range [m]'); 
ylabel('Rxe'); title('Q12: Matched filter by cross-corrrelation RX/TX');
hold off

% fftxe=fft(Rxe)
% fs=1/Ts; % reference frequency
% 
% Sn_shift=fftshift(fftxe);
% fshift=(-K:K-2).*(fs/2*K); %time shifting
% pwrshift=(1/(2*K*fs)).*abs(Sn_shift).^2; % Sn
% figure(18); plot(fshift,pwrshift); grid on; box on; xlabel('f [Hz]'); 
% ylabel('Sxe[W] centered');title('Q12: PSD Sxe(f) representation');
% SndB_shift=10*log10(pwrshift); %Sn in dbW
% figure(19); plot(fshift,SndB_shift); grid on; box on; xlabel('f [Hz]'); 
% ylabel('G(Sxe)[dbW] entered');title('Q12: PSD Sxe(f) representation');





%% Question 13

% 13(a)

B=beta*T
time_R=1/B
delta_R=c/(2*B)

% 13(b)















