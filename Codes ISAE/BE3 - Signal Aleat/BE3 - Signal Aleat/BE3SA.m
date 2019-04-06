%% BE3 - Random signal Processing
% PC-6
% Eduardo DADALTO CAMARA GOMES
% My-Loan DANG
% Gabriel GOMES PASSOS

%%
clear all; close all; clc;
%% Question 4
Tr = 1e-3; % pulse repetition interval (defined: 1ms)
T = Tr/10; % Pulse duration
Ts = T/100; % Time samplig
sig2 = 1; % DEFINED SUJET: dBW

t = 0:Ts:Tr-Ts; % Domain of discrete-time
K = floor(Tr/Ts); % Nearest integer towards minus infinity of Tr/ts

kT = floor(T/Ts);
vec1 = ones(1,kT+1);

% taille vecteur zeros
n0 = K-kT;
vec0 = zeros(1,n0-1);
e = [vec1 vec0];

%generate e
e=rectangularPulse(0,T,t);

% representing e
figure(1); plot(t,e); grid on; box on; xlabel('t[s]'); ylabel('e(t)');
title('Q4: Signal e(t)');

%% Question 5 : SoI
tau = (Tr-2*T)*rand(1) + T;
ktau = floor(tau/Ts);

a = [zeros(1,ktau-1), e(1,1:end - ktau + 1)];
re = rand(1);
im = rand(1);
alfa = (re + 1i*im);
SoI = alfa*a;

% representing SoI : Re(alfa*a(t))
figure(2); plot(t,SoI), grid on; box on; xlabel('t[s]'); 
ylabel('Re{\alpha a(t)}'); title('Q5: SoI Signal Re{\alpha a(t)}');
% alternativce representation of SoI : Re(alpha*a(t)) x Im(alpha*a(t))
figure(3); plot3(t,real(SoI),imag(SoI)); grid on; box on;
xlabel('Re{\alpha a(t)}');  ylabel('Im{\alpha a(t)}'); 
title('Q5: SoI Signal Re(\alpha a(t)) x Im(\alpha a(t)))');
%% Question 6
mu = 0;

nr = sqrt(sig2/2).*randn(1,K);
ni = sqrt(sig2/2).*randn(1,K);
% Proposed method in the preliminary work
n = nr + 1j*ni;

% Generating x
x = SoI + n;

% Representating x
figure(4); plot3(t,real(x),imag(x)); grid on; box on; xlabel('t[s]'); 
ylabel('Re(\alpha a(t))'); zlabel('Im(\alpha a(t))');
title('Q6: SoI Signal Re(\alpha a(t))');

%% Question 7
% Verifying mean
mnr = mean(nr);
mni = mean(ni);
% Verifying variance
varnr = var(nr);
varni = var(ni);

%verifying PDF
PDFnr=normpdf(nr,mu,sig2);
figure(5); scatter(nr,PDFnr); grid on; box on; xlabel('nr'); 
ylabel('PDF(nr)');title('Q7: PDF of nr');

PDFni=normpdf(ni,mu,sig2);
figure(6); scatter(ni,PDFni); grid on; box on; xlabel('ni'); 
ylabel('PDF(ni)');title('Q7: PDF of ni');

% verifying histogram
figure(7);histogram(nr); grid on; box on; xlabel('value'); 
ylabel('# realization');title('Q7: Histogram of nr');
figure(8); histogram(ni); grid on; box on; xlabel('value'); 
ylabel('# realization');title('Q7: Histogram of ni');

FFTnr = fft(nr);
FFTni = fft(ni);

FFTn=fft(n); % Fast Fourier Transformer 
fs=1/Ts; % reference frequency
Sn=(1/(K*fs)).*(abs(FFTn)).^2; %periodogram method 
f=linspace(0,fs,K); % discrete-frequency for representation 
figure(9); plot(f,Sn); grid on; box on; xlabel('f [Hz]'); 
ylabel('Sn [W/Hz]');title('Q7: PSD Sn(f) representation');

Sn_shift=fftshift(FFTnr);
fshift=(-K/2:K/2-1).*(fs/K); %time shifting
pwrshift=(1/(K*fs)).*abs(Sn_shift).^2; % Sn
figure(10); plot(fshift,pwrshift); grid on; box on; xlabel('f [Hz]'); 
ylabel('Sn[W] centered');title('Q7: PSD Sn(f) representation');
SndB_shift=10*log10(pwrshift); %Sn in dbW
figure(11); plot(fshift,SndB_shift); grid on; box on; xlabel('f [Hz]'); 
ylabel('G(Sn)[dbW] entered');title('Q7: PSD Sn(f) representation');

%% Question 8
% Implementing the matched filter
% Crross correlation between the emitted signal and the received signal
[Rxe,lag] = xcorr(x,e); %R(x(t1),e(t2))
yk = Rxe;

% Representation of cross correlation function
figure(12); hold on; plot(lag*Ts,Rxe); 
grid on; box on; xlabel('\tau [s]'); 
ylabel('Rxe'); title('Q8: Matched filter by cross-corrrelation RX/TX');

% Calculate Distance between targets
c = 3e8; % Light speed
R = c/2*lag*Ts; % Range calculation - Transforms time in distance
[peaks, plocs] = findpeaks(real(Rxe*Ts)); % Find all the peaks in the vector
[maxpeak,pos] = max(peaks); % Find the peak plus intense

range = R(plocs(pos)); % Positions in the axis X 

% Second representation of the cross correlation function with the distance
% between the target and the radar
figure(13);hold on;plot(R,Rxe*Ts); scatter(range,maxpeak,'or');
grid on; box on; xlabel('Range [m]'); 
leg1=legend('Rxe',...
                    ['Peak of signal: ','Range=',num2str(range),'m'],...
                'Location','northwest' );
            title(leg1,'Legend')
            legend('boxoff')
ylabel('Rxe'); title('Q8: Matched filter by cross-corrrelation RX/TX');
hold off;

delta_R=c*T/2; % Is equal approximately to half of the size of the base of 
% the triangle of the graph

%% Question 12

beta=2e8;% user defined
% generating e(t)
e = exp((1i*2*pi*beta/2).*t.^2).*rectangularPulse(0,T,t);

figure(14); plot3(t,real(e),imag(e)); grid on; box on; xlabel('t [s]'); 
ylabel('Re(e(t))');zlabel('Im(e(t))'); title('Q12: e(t) "chirp" emitted');

a = [zeros(1,ktau-1), e(1,1:end - ktau + 1)];
re = rand(1);
im = rand(1);
alfa = (re + 1i*im);
SoI = alfa*a;

x = SoI + n;

figure(15); plot3(t,real(x),imag(x)); grid on; box on; xlabel('t [s]'); 
ylabel('Re(e(t))');zlabel('Im(e(t))'); title('Q12: e(t) "chirp" received');

[Rxe,lag] = xcorr(x,e); %R(x(t1),e(t2))
yk = Rxe;

% Representation of cross correlation function
figure(16); hold on; plot(lag*Ts,Rxe); 
grid on; box on; xlabel('\tau [s]'); 
ylabel('Rxe'); title('Q12: Matched filter by cross-corrrelation RX/TX');

% Calculate Distance between targets
c = 3e8; % Light speed
R = c/2*lag*Ts; % Range calculation - Transforms time in distance
[peaks, plocs] = findpeaks(real(Rxe*Ts)); % Find all the peaks in the vector
[maxpeak,pos] = max(peaks); % Find the peak plus intense

range = R(plocs(pos)); % Positions in the axis X 

% Second representation of the cross correlation function with the distance
% between the target and the radar
figure(17);hold on;plot(R,Rxe*Ts); scatter(range,maxpeak,'or');
grid on; box on; xlabel('Range [m]'); 
leg1=legend('Rxe',...
                    ['Peak of signal: ','Range=',num2str(range),'m'],...
                'Location','northwest' );
            title(leg1,'Legend')
            legend('boxoff')
ylabel('Rxe'); title('Q12: Matched filter by cross-corrrelation RX/TX');
hold off;
figure (18);
hold on;
plot(t,e)
plot(t,x)

%% Question 13
B=beta*T;
[Re,lag] = xcorr(e,e); %R(e(t1),e(t2))
sincf = 100*sinc(pi*B.*lag*Ts)

figure(19); hold on; plot(lag*Ts,Re); plot(lag*Ts,sincf);hold off;
grid on; box on; xlabel('\tau [s]'); 
ylabel('Rxe'); title('Q13: Matched filter by cross-corrrelation RX/TX');


