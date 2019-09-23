% requirement: JSONLab: https://cn.mathworks.com/matlabcentral/fileexchange/33381-jsonlab--a-toolbox-to-encode-decode-json-files

format long
addpath('D:\Users\foamliu\code\jsonlab');
facescrub_cmc_file = 'D:\Users\foamliu\code\InsightFace-v2\megaface\results\cmc_facescrub_megaface_0_1000000_1.json'
facescrub_cmc_json = loadjson(fileread(facescrub_cmc_file));
facescrub_cmc_json


figure(1);
semilogx(facescrub_cmc_json.cmc(1,:)+1,facescrub_cmc_json.cmc(2,:)*100,'LineWidth',2);
title(['Identification @ 1e6 distractors = ' num2str(facescrub_cmc_json.cmc(2,:)(1))]);
xlabel('Rank');
ylabel('Identification Rate %');
%ylim([0 100]);
grid on;
box on;
hold on;

facescrub_cmc_json.roc(1,:)

figure(2);
%semilogx(facescrub_cmc_json.roc(1,:),facescrub_cmc_json.roc(2,:),'LineWidth',2);
xdata=[0.0,			1.028475438147325e-08,			3.085426314441975e-08,			2.262646034978388e-07,			9.976212140827556e-07,			6.757083610864356e-06,			0.0001522452221252024,			1.0		],
ydata=[0.9281878471374512,			0.9383026361465454,			0.9504063129425049,			0.9605637192726135,			0.9707068800926208,			0.9807222485542297,			0.9907233715057373,			1.0]
semilogx(xdata,ydata,'LineWidth',2);
%semilogx(facescrub_cmc_json.roc{1},facescrub_cmc_json.roc{2},'LineWidth',2);
title(['Verification @ 1e-6 = ' num2str(interp1(xdata, ydata, 1e-6))]);
xlim([1e-6 1]);
%ylim([0 1]);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
grid on;
box on;
hold on;