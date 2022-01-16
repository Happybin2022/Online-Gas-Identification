datas = readmatrix("C:\Users\Happybin\Desktop\连续传感器数据在线监测\dataset\ethylene_CO_1Hz.csv");

% 去掉前10个数据
[time, sensor] = size(datas);
datas = datas(10:time, 2:sensor);
[time, sensor] = size(datas);
Time = 1:time;

figure(1);
plot(1:10000, datas(1:10000, 1), 1:10000, datas(1:10000, 2))

% 浓度平滑
current_conv_1 = 0;
current_conv_2 = 0;
next_conv_1 = 0;
next_conv_2 = 0;

length = 24;
smooth_flag_1 = length + 1;
smooth_flag_2 = length + 1;
delta_conv_1 = 0;
delta_conv_2 = 0;

for i=1:(time-1)
    % 处理浓度1
    if smooth_flag_1 == length + 1
        current_conv_1 = datas(i, 1);
        next_conv_1 = datas(i+1, 1);
        if current_conv_1 == next_conv_1
            datas(i, 1) = datas(i, 1);
        else
            smooth_flag_1 = 1;
            dalta_conv_1 = (next_conv_1 - current_conv_1) / length;
        end
    else
        temp_1 = current_conv_1 + smooth_flag_1 * dalta_conv_1;
        datas(i, 1) = temp_1;
        smooth_flag_1 = smooth_flag_1 + 1;
    end
    
    % 处理浓度2
    if smooth_flag_2 == length + 1
        current_conv_2 = datas(i, 2);
        next_conv_2 = datas(i+1, 2);
        if current_conv_2 == next_conv_2
            datas(i, 2) = datas(i, 2);
        else
            smooth_flag_2 = 0;
            dalta_conv_2 = (next_conv_2 - current_conv_2) / length;
        end
    else
        datas(i, 2) = current_conv_2 + smooth_flag_2 * dalta_conv_2;
        smooth_flag_2 = smooth_flag_2 + 1;
    end     
end
figure(2);
plot(1:10000, datas(1:10000, 1), 1:10000, datas(1:10000, 2))
Time = Time.';
datas = [Time, datas];
writematrix(datas, "C:\Users\Happybin\Desktop\连续传感器数据在线监测\dataset\ethylene_CO_1Hz_smooth.csv");
