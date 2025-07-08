% 实验数据解析与可视化
clear; clc;

% 1. 原始数据输入
data_str = ['0:58/124; 0.1:58/124; 0.2:58/124; 0.3:58/124;',...
            '0.4:58/124;0.5:58/124; 0.6:58/124; 0.7:58/126;',...
            '0.8:58/131; 0.81:68/162;0.82:68/162; 0.83:68/162;0.84:68/165;0.85:72,178; 0.87:72,181; 0.88:72,182;',...
            '0.89:deadlock;0.9:deadlock;'];

% 2. 数据解析处理
data_points = strsplit(data_str, ';');
a_values = [];      % 有效参数值
states = [];        % 可达状态数
transitions = [];   % 变迁数
deadlock_a = [];    % 死锁参数值

for i = 1:length(data_points)
    point = strtrim(data_points{i});
    if isempty(point), continue; end
    
    % 分割参数与数值
    parts = strsplit(point, ':');
    if numel(parts) ~= 2, continue; end
    
    % 解析参数a
    a = str2double(parts{1});
    if isnan(a), continue; end
    
    % 处理deadlock情况
    if strcmpi(parts{2}, 'deadlock')
        deadlock_a = [deadlock_a; a];
        continue;
    end
    
    % 解析状态数和变迁数（兼容/和,分隔符）
    nums = regexp(parts{2}, '(\d+)', 'tokens');
    if numel(nums) ~= 2, continue; end
    state = str2double(nums{1}{1});
    transition = str2double(nums{2}{1});
    
    % 记录有效数据
    a_values = [a_values; a];
    states = [states; state];
    transitions = [transitions; transition];
end

% 3. 可视化绘制
figure('Color','w','Position',[100 100 800 400])
hold on

% 绘制状态数曲线
plot(a_values, states, 'b-o', 'LineWidth',1.5,...
    'MarkerSize',8, 'MarkerFaceColor','b',...
    'DisplayName','可达状态数')

% 绘制变迁数曲线
plot(a_values, transitions, 'r--s', 'LineWidth',1.5,...
    'MarkerSize',8, 'MarkerFaceColor','r',...
    'DisplayName','变迁数')

% 标注死锁位置
y_max = max([states; transitions])*1.1;
scatter(deadlock_a, repmat(y_max,size(deadlock_a)),...
    'kv', 'LineWidth',1.5, 'MarkerSize',12,...
    'DisplayName','死锁发生点')

% 图形修饰
xlabel('参数 a','FontSize',12,'FontWeight','bold')
ylabel('数量统计','FontSize',12,'FontWeight','bold')
title('系统状态随参数a的变化趋势','FontSize',14,'FontWeight','bold')
legend('Location','northwest')
grid on
box on

% 设置坐标轴范围
xlim([0, 0.9])
set(gca,'XTick',sort([0:0.1:0.9 deadlock_a']),...
        'XTickLabelRotation',45)
set(gca,'FontSize',11)

% 4. 添加死锁标注
for a = deadlock_a'
    text(a, y_max*0.95, 'Deadlock',...
        'Rotation',90, 'VerticalAlignment','top',...
        'FontSize',10, 'Color','k', 'FontWeight','bold')
end