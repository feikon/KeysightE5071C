"""
June 2022
@author: Mathieu Couillard

Driver for Keysight K5071C Vector Network Analyzer
"""

import numpy as np
import pyvisa as visa

from time import sleep


def format_num(arg, units=1, limits=(-float('inf'),float('inf'))) -> str:
    """
    格式化数字参数为SCPI命令格式

    参数:
        arg: 要格式化的数字，可以是数字、None或'?'
        units: 单位转换因子，默认为1
        limits: 数值范围限制，默认为无限制

    返回:
        str: 格式化后的字符串

    使用实例:
        >>> format_num(1.5, units=1e9)  # 将1.5转换为1.5GHz
        ' 1500000000'
        >>> format_num('?')  # 查询参数
        '?'
    """
    if arg == None or arg == '?':
        return '?'
    else:
        # TODO: Make dictionary of units
        arg = float(arg)*units
        if arg%1 == 0:
            arg = int(arg)
        if limits[0]<=arg<=limits[1]:
            return ' ' + str(arg)
        else:
            raise Exception("OutOfRangeException: Value must be between {} and {}.".format(limits[0], limits[1]))

def format_from_dict(arg, arg_dict) -> str:
    """
    从字典中格式化参数

    参数:
        arg: 输入参数
        arg_dict: 参数映射字典

    返回:
        str: 格式化后的字符串

    使用实例:
        >>> options = {'on': ' ON', 'off': ' OFF'}
        >>> format_from_dict('on', options)
        ' ON'
    """
    if arg == None:
        arg = '?'
    arg = str(arg).lower()
    try:
        return arg_dict[arg]
    except:
        print("InvalidInputError: Argument must be : {}".format(list(arg_dict.keys())))
        return '?' # FIXME: There should be a better way to handle this error with querying the device.

class E5071C:
    """
    Keysight E5071C矢量网络分析仪驱动类

    该类提供了控制Keysight E5071C VNA的完整接口，
    包括频率设置、数据读取、平均、触发等功能。

    使用实例:
        >>> vna = E5071C("TCPIP::192.168.1.100::INSTR")
        >>> vna.freq_start(1e9)  # 设置起始频率为1GHz
        >>> vna.freq_stop(2e9)   # 设置终止频率为2GHz
        >>> data = vna.read_all_traces()  # 读取所有迹近数据
    """
    def __init__(self, address, configs="", visa_backend=None, verbatim=False):
        """
        初始化E5071C实例

        参数:
            address: VISA设备地址，如"TCPIP::192.168.1.100::INSTR"
            configs: 配置字符串（备用）
            visa_backend: VISA后端，默认None使用默认后端
            verbatim: 是否打印每个发送的命令，默认False

        使用实例:
            >>> vna = E5071C("TCPIP::192.168.1.100::INSTR", verbatim=True)
        """
        if visa_backend==None:
            self._inst = visa.ResourceManager().open_resource(address)
        else:
            self._inst = visa.ResourceManager(visa_backend).open_resource(address)
        self.verbatim = verbatim
        identity = self.identify()
        print("Identity: {}".format(identity))
        if "E5071C" not in identity:
            Exception("WARNING: The device:{} is not a E5071C vector network analyzer."
                      "\nSome commands may not work.".format(address))

        self.verbatim = verbatim  # Print every command before sending

    ########################################
    # Selecting channel and trace
    ########################################
    def traces_number(self, num=None, chan=""):
        """
        设置或查询迹近数量

        参数:
            num: 要设置的迹近数量，None表示查询
            chan: 通道编号，默认为当前活动通道

        返回:
            如果num是None，返回当前迹近数量；否则返回命令确认

        使用实例:
            >>> vna.traces_number(2)  # 设置为2条迹近
            >>> vna.traces_number()   # 查询当前迹近数量
        """
        if num != None:
            num = " " + str(num)
        elif num == None:
            num = "?"
        return self._com(":CALC{}:PAR:COUN{}".format(chan, num))

    def displayed_channels(self, chans='?'):
        """
        设置或查询显示的通道

        参数:
            chans: 要显示的通道组合，支持'1', '12', '123', '1234', '123456'

        返回:
            命令确认或当前设置

        使用实例:
            >>> vna.displayed_channels('12')  # 显示通道1和2
            >>> vna.displayed_channels()      # 查询当前设置
        """
        options = {'1': ' D1',
                   '12': ' D1_2',
                   '13': ' D1_3',
                   '123': ' D1_2_3',
                   '1234': ' D1_2_3_4',
                   '123456': ' D1_2_3_4_5_6',
                   '?': '?'
                   }
        chans = format_from_dict(chans, options)
        return self._com(":DISP:SPL{}".format(chans))

    def active_chan(self, chan=None):
        """
        设置或查询活动通道

        参数:
            chan: 通道编号，None表示查询

        返回:
            如果chan为None，返回当前活动通道；否则返回命令确认

        使用实例:
            >>> vna.active_chan(1)  # 设置通道1为活动通道
            >>> vna.active_chan()   # 查询当前活动通道
        """
        chan = format_num(chan)
        if chan == '?':
            return self._com(':SERV:CHAN:ACT?')
        else:
            return self._com(":DISP:WIND{}:ACT".format(chan))

    def active_trace(self, trace=None, chan=""):
        """
        设置或查询活动迹近

        参数:
            trace: 迹近编号，None表示查询
            chan: 通道编号

        返回:
            如果trace为None，返回当前活动迹近；否则返回命令确认

        使用实例:
            >>> vna.active_trace(1)  # 设置迹近1为活动迹近
            >>> vna.active_trace()   # 查询当前活动迹近
        """
        trace = format_num(trace)
        if trace == '?':
            return self._com(":SERV:CHAN{}:TRAC:ACT?".format(chan))
        else:
            return self._com(':CALC{}:PAR{}:SEL'.format(chan, trace))

    ########################################
    # Averaging
    ########################################

    def average_reset(self, chan=""):
        """
        重置平均统计

        参数:
            chan: 通道编号

        返回:
            命令确认

        使用实例:
            >>> vna.average_reset()  # 重置平均统计
        """
        return self._com(":SENS{}:AVER:CLE".format(chan))

    def average_count(self, count=None, chan=""):
        """
        设置或查询平均次数

        参数:
            count: 平均次数，None表示查询
            chan: 通道编号

        返回:
            如果count为None，返回当前平均次数；否则返回命令确认

        使用实例:
            >>> vna.average_count(10)  # 设置平均次数为10
            >>> vna.average_count()    # 查询当前平均次数
        """
        count = format_num(count)
        return self._com(":SENS{}:AVER:COUN{}".format(chan, count))

    def average_state(self, state=None, chan=""):
        """
        设置或查询平均状态

        参数:
            state: 平均状态，'on'/'true'/1开启，'off'/'false'/0关闭，None查询
            chan: 通道编号

        返回:
            如果state为None，返回当前平均状态；否则返回命令确认

        使用实例:
            >>> vna.average_state('on')   # 开启平均
            >>> vna.average_state('off')  # 关闭平均
            >>> vna.average_state()       # 查询平均状态
        """
        options = {'on': " on", '1': ' 1', 'true': ' on',
                   'off': ' off', '0': ' 0', 'false': ' 0',
                   '?': '?'
                   }
        state = format_from_dict(state, options)
        return self._com(":SENS{}:AVER:STAT{}".format(chan, state))

    ########################################
    # Frequency axis
    ########################################
    # TODO: Make argument to choose units from a dictionary and make the default GHz
    def freq_start(self, freq=None, chan=""):
        """
        设置或查询起始频率

        参数:
            freq: 起始频率（Hz），None表示查询
            chan: 通道编号

        返回:
            如果freq为None，返回当前起始频率；否则返回命令确认

        使用实例:
            >>> vna.freq_start(1e9)    # 设置起始频率为1GHz
            >>> vna.freq_start()       # 查询当前起始频率
        """
        freq = format_num(freq, 1)
        return self._com(":SENS{}:FREQ:STAR{}".format(chan, freq))

    def freq_stop(self, freq=None, chan=""):
        """
        设置或查询终止频率

        参数:
            freq: 终止频率（Hz），None表示查询
            chan: 通道编号

        返回:
            如果freq为None，返回当前终止频率；否则返回命令确认

        使用实例:
            >>> vna.freq_stop(2e9)     # 设置终止频率为2GHz
            >>> vna.freq_stop()        # 查询当前终止频率
        """
        freq = format_num(freq, 1)
        return self._com(":SENS{}:FREQ:STOP{}".format(chan, freq))

    def freq_center(self, freq=None, chan=""):
        """
        设置或查询中心频率

        参数:
            freq: 中心频率（Hz），None表示查询
            chan: 通道编号

        返回:
            如果freq为None，返回当前中心频率；否则返回命令确认

        使用实例:
            >>> vna.freq_center(1.5e9)  # 设置中心频率为1.5GHz
            >>> vna.freq_center()       # 查询当前中心频率
        """
        freq = format_num(freq, 1)
        return self._com(":SENS{}:FREQ:CENT{}".format(chan, freq))

    def freq_span(self, freq=None, chan=""):
        """
        设置或查询频率范围

        参数:
            freq: 频率范围（Hz），None表示查询
            chan: 通道编号

        返回:
            如果freq为None，返回当前频率范围；否则返回命令确认

        使用实例:
            >>> vna.freq_span(1e9)     # 设置频率范围为1GHz
            >>> vna.freq_span()        # 查询当前频率范围
        """
        freq = format_num(freq, 1)
        return self._com(":SENS{}:FREQ:SPAN{}".format(chan, freq))

    def points(self, points=None, chan=""):
        """
        设置或查询测量点数

        参数:
            points: 测量点数，None表示查询
            chan: 通道编号

        返回:
            如果points为None，返回当前点数；否则返回命令确认

        使用实例:
            >>> vna.points(1001)       # 设置测量点数为1001
            >>> vna.points()           # 查询当前点数
        """
        points = format_num(points)
        return self._com(":SENS{}:SWE:POIN{}".format(chan, points))

    def ifbw(self, bandwidth=None, chan=""):
        """
        设置或查询中频带宽

        参数:
            bandwidth: 中频带宽（Hz），None表示查询
            chan: 通道编号

        返回:
            如果bandwidth为None，返回当前带宽；否则返回命令确认

        使用实例:
            >>> vna.ifbw(1000)         # 设置中频带宽为1kHz
            >>> vna.ifbw()             # 查询当前带宽
        """
        bandwidth = format_num(bandwidth)
        return self._com(":SENS{}:BAND:RES{}".format(chan, bandwidth))

    def bandwidth(self, bandwidth=None, chan=""):
        """
        设置或查询带宽（ifbw的别名）

        参数:
            bandwidth: 带宽（Hz），None表示查询
            chan: 通道编号

        返回:
            调用ifbw方法的结果

        使用实例:
            >>> vna.bandwidth(1000)    # 设置带宽为1kHz
        """
        return self.ifbw(bandwidth, chan)

    ########################################
    # Response
    ########################################

    def format_trace(self, trace_format=None, chan=""):
        """
        设置或查询迹近显示格式

        参数:
            trace_format: 迹近格式，支持'mlog'(对数幅度)、'phase'(相位)、
                         'lin_mag'(线性幅度)、'real'(实部)、'imag'(虚部)等
            chan: 通道编号

        返回:
            如果trace_format为None，返回当前格式；否则返回命令确认

        使用实例:
            >>> vna.format_trace('mlog')     # 设置为对数幅度格式
            >>> vna.format_trace('phase')    # 设置为相位格式
            >>> vna.format_trace()           # 查询当前格式
        """
        trace_formats = {'mlog': ' MLOG',
                         'phase': ' PHAS',
                         'lin_mag': ' MLIN',
                         'real': ' REAL',
                         'imag': " IMAG",
                         'extend_phase': ' UPH',
                         'uph': ' UPH',
                         'positive_phase': ' PPH',
                         'pph': ' PPH',
                         'polar_linear': ' PLIN',
                         'plin': ' PLIN',
                         'polar_log': ' PLOG',
                         'plog': ' PLOG',
                         'real_imag': ' POL',
                         '?': '?'}
        trace_format = format_from_dict(trace_format,trace_formats)
        return self._com(':CALC{}:SEL:FORM{}'.format(chan, trace_format))

    ########################################
    # Output
    ########################################

    def delay(self, delay=None, chan=""):
        """
        设置或查询电气延迟

        参数:
            delay: 延迟时间（秒），None表示查询
            chan: 通道编号

        返回:
            如果delay为None，返回当前延迟；否则返回命令确认

        使用实例:
            >>> vna.delay(1e-9)        # 设置延迟为1ns
            >>> vna.delay()            # 查询当前延迟
        """
        delay = format_num(delay)
        return self._com(":CALC{}:CORR:EDEL:TIME{}".format(chan, delay))

    def phase_offset(self, phase=None, chan=""):
        """
        设置或查询相位偏移

        参数:
            phase: 相位偏移（度），None表示查询
            chan: 通道编号

        返回:
            如果phase为None，返回当前相位偏移；否则返回命令确认

        使用实例:
            >>> vna.phase_offset(90)   # 设置相位偏移为90度
            >>> vna.phase_offset()     # 查询当前相位偏移
        """
        phase = format_num(phase)
        return self._com(":CALC{}:CORR:OFFS:PHAS{}".format(chan, phase))

    def power(self, power=None, source=''):
        """
        设置或查询输出功率

        参数:
            power: 输出功率（dBm），None表示查询
            source: 信号源编号

        返回:
            如果power为None，返回当前功率；否则返回命令确认

        使用实例:
            >>> vna.power(-10)         # 设置输出功率为-10dBm
            >>> vna.power()            # 查询当前功率
        """
        power = format_num(power)
        return self._com(':SOUR{}:POW{}'.format(source, power))

    def output(self, out=None):
        """
        设置或查询RF输出状态

        参数:
            out: 输出状态，'on'/'true'/1开启，'off'/'false'/0关闭，None查询

        返回:
            如果out为None，返回当前RF输出状态；否则返回命令确认

        使用实例:
            >>> vna.output('on')        # 开启RF输出
            >>> vna.output('off')       # 关闭RF输出
            >>> vna.output()            # 查询RF输出状态
        """
        options = {'true': ' 1',
                   'on': ' 1',
                   '1': ' 1',
                   'false': ' 0',
                   'off': ' 0',
                   '0': ' 0',
                   '?': '?'
                   }
        out = format_from_dict(out, options)
        return self._com(":OUTP{}".format(out))

    def sweep_type(self, sweep_type=None, chan=""):
        """
        设置或查询扫描类型

        参数:
            sweep_type: 扫描类型，'linear'(线性)、'log'(对数)、'segmented'(分段)等
            chan: 通道编号

        返回:
            如果sweep_type为None，返回当前扫描类型；否则返回命令确认

        使用实例:
            >>> vna.sweep_type('linear') # 设置为线性扫描
            >>> vna.sweep_type('log')    # 设置为对数扫描
            >>> vna.sweep_type()         # 查询当前扫描类型
        """
        sweep_types = {'linear': ' LIN',
                       'lin': ' LIN',
                       'log': ' LOG',
                       'segmented': ' SEG',
                       'power': ' POW',
                       '?': '?'
                       }
        sweep_type = format_from_dict(sweep_type, sweep_types)
        return self._com(':SENS{}:SWE:TYPE{}'.format(chan, sweep_type))

    def s_par(self, s_par=None, trace="", chan=""):
        """
        设置或查询S参数类型

        参数:
            s_par: S参数类型，如'S11', 'S12', 'S21', 'S22'等
            trace: 迹近编号
            chan: 通道编号

        返回:
            如果s_par为None，返回当前S参数；否则返回命令确认

        使用实例:
            >>> vna.s_par('S11')       # 设置为S11参数
            >>> vna.s_par('S21')       # 设置为S21参数
            >>> vna.s_par()            # 查询当前S参数
        """
        options = {'s11': ' S11', 's12': ' S12', 's13': ' S13', 's14': ' S14',
                   's21': ' S21', 's22': ' S22', 's23': ' S23', 's24': ' S24',
                   's31': ' S31', 's32': ' S32', 's33': ' S33', 's34': ' S34',
                   's41': ' S41', 's42': ' S42', 's43': ' S43', 's44': ' S44',
                   '?': '?'
                   }
        s_par = format_from_dict(s_par,options)
        if s_par in options:
            return self._com(':CALC{}:PAR{}:DEF{}'.format(chan, trace, s_par))

    ########################################
    # Trigger
    ########################################
    def trigger_source(self, source=None):
        """
        设置或查询触发源

        参数:
            source: 触发源类型，'internal'(内部)、'external'(外部)、
                   'manual'(手动)、'bus'(总线)

        返回:
            如果source为None，返回当前触发源；否则返回命令确认

        使用实例:
            >>> vna.trigger_source('bus')      # 设置为总线触发
            >>> vna.trigger_source('internal') # 设置为内部触发
            >>> vna.trigger_source()           # 查询当前触发源
        """
        sources = {"internal": " INT",
                   "external": " EXT",
                   "manual": " MAN",
                   "bus": " BUS",
                   "?": "?"
                   }
        source = format_from_dict(source, sources)
        return self._com(":TRIG:SOUR{}".format(source))


    def trigger_initiate(self, state=None, chan=""):
        """
        设置触发初始化状态

        参数:
            state: 触发状态，'cont'(连续)、'hold'(保持)、'single'(单次)
            chan: 通道编号

        返回:
            命令确认

        使用实例:
            >>> vna.trigger_initiate('cont')   # 设置为连续触发
            >>> vna.trigger_initiate('single') # 设置为单次触发
            >>> vna.trigger_initiate('hold')   # 设置为保持状态
        """
        options = {"cont": ":CONT ON",
                   "hold": ":CONT OFF",
                   "single": "",
                   "?": "?"
                   }
        state = format_from_dict(state, options)
        return self._com('INIT{}{}'.format(chan, state))

    def trigger_now(self):
        """
        立即执行一次触发测量

        该方法会发送单次触发命令，并等待测量完成。
        等待时间基于平均次数和扫描时间计算。

        返回:
            包含测量完成信息的字符串

        使用实例:
            >>> vna.trigger_now()  # 立即执行一次测量
        """
        if self.average_state() == 1:
            average_count = self.average_count()
        else:
            average_count = 1

        self._com(":TRIG:SING")
        sweep_time = float(self.get_sweep_time())
        sleep(int(average_count) * sweep_time)
        return 'Sent: :TRIG:SING \nMeasuremet complete {}'.format(self.operation_complete())


    def trigger_averaging(self, averaging=None):
        """
        设置或查询触发平均状态

        参数:
            averaging: 平均状态，'on'/'true'/1开启，'off'/'false'/0关闭

        返回:
            如果averaging为None，返回当前状态；否则返回命令确认

        使用实例:
            >>> vna.trigger_averaging('on')  # 开启触发平均
            >>> vna.trigger_averaging('off') # 关闭触发平均
            >>> vna.trigger_averaging()      # 查询当前状态
        """
        options = {"on": " ON", "1": " 1", "true": " 1",
                   "off": " OFF", "0": " 0", "false": " 0",
                   "?": "?"
                   }
        averaging = format_from_dict(averaging, options)
        return self._com(":TRIG:SEQ:AVER{}".format(averaging))

    ########################################
    # reading data
    ########################################

    def format_data(self, form=None):
        """
        设置或查询数据格式

        参数:
            form: 数据格式，'ascii'(ASCII格式)、'real'(64位实数)、'real32'(32位实数)

        返回:
            如果form为None，返回当前数据格式；否则返回命令确认

        使用实例:
            >>> vna.format_data('real')  # 设置为二进制实数格式
            >>> vna.format_data('ascii') # 设置为ASCII格式
            >>> vna.format_data()        # 查询当前数据格式
        """
        formats = {'ascii': ' ASC',
                   'asc': ' ASC',
                   'real': ' REAL',
                   'real32': ' REAL32',
                   '?': '?'
                   }
        form = format_from_dict(form, formats)
        return self._com(':FORMat:DATA{}'.format(form))

    def read_freq(self):
        """
        读取频率轴数据

        该方法会自动设置数据格式为二进制实数，读取数据后恢复为ASCII格式。

        返回:
            numpy数组: 频率轴数据（Hz）

        使用实例:
            >>> freq_axis = vna.read_freq()
            >>> print(f"频率范围: {freq_axis[0]:.2e} - {freq_axis[-1]:.2e} Hz")
        """
        self.format_data('real')
        data = self._com_binary(':CALC:SEL:DATA:XAXis?')
        self.format_data('ascii')
        return data

    def read_trace(self, trace=None):
        """
        读取指定迹近的复数数据

        该方法读取指定迹近的复数数据，包括实部和虚部。

        参数:
            trace: 迹近编号，None表示使用当前活动迹近

        返回:
            tuple: (实部数据, 虚部数据)，两个 numpy 数组

        使用实例:
            >>> real_part, imag_part = vna.read_trace(1)
            >>> magnitude = np.sqrt(real_part**2 + imag_part**2)
            >>> phase = np.arctan2(imag_part, real_part) * 180 / np.pi
        """
        if trace==None:
            trace = '?'
        self.format_data('real')
        data = self._com_binary(':CALC:TRACe{}:DATA:FDATa?'.format(trace))
        self.format_data('ascii')

        return data[0::2], data[1::2]

    def read_all_traces(self):
        """
        读取当前活动通道的所有迹近数据

        该方法读取当前活动通道的所有迹近数据，包括频率轴和每条迹近的复数数据。

        返回:
            numpy数组: 形状为(2*traces+1, points)
                        - 第0行: 频率轴数据
                        - 奇数行: 各迹近的实部数据
                        - 偶数行: 各迹近的虚部数据

        使用实例:
            >>> data = vna.read_all_traces()
            >>> freq = data[0]  # 频率轴
            >>> trace1_real = data[1]  # 第1条迹近实部
            >>> trace1_imag = data[2]  # 第1条迹近虚部
            >>> trace2_real = data[3]  # 第2条迹近实部
            >>> trace2_imag = data[4]  # 第2条迹近虚部
        """
        # read the x axis and all traces of the active channel
        traces = int(self.traces_number())
        points = int(self.points())

        data = np.empty((2*traces+1, points))
        self.format_data('real')

        data[0] = self._com_binary(':CALC:SEL:DATA:XAXis?')
        for trace in range(traces):
            raw_data = self._com_binary(':CALC:TRACe{}:DATA:FDATa?'.format(trace+1))
            data[2*trace + 1] = raw_data[0::2]
            data[2*trace + 2] = raw_data[1::2]

        self.format_data('ascii')
        return data


    def close(self):
        """
        关闭VISA连接

        使用实例:
            >>> vna.close()  # 关闭与仪器的连接
        """
        self._inst.close()

    def identify(self):
        """
        查询仪器身份信息

        返回:
            str: 仪器身份字符串，包括制造商、型号、序列号等

        使用实例:
            >>> identity = vna.identify()
            >>> print(identity)
        """
        return self._com("*IDN?")

    def idn(self):
        """
        查询仪器身份信息（identify的别名）

        返回:
            str: 仪器身份字符串

        使用实例:
            >>> identity = vna.idn()
        """
        return self._com("*IDN?")

    def reset(self):
        """
        重置仪器到默认状态

        返回:
            str: 命令确认

        使用实例:
            >>> vna.reset()  # 重置仪器
        """
        return self._com('*RST')

    def rst(self):
        """
        重置仪器到默认状态（reset的别名）

        返回:
            str: 命令确认

        使用实例:
            >>> vna.rst()  # 重置仪器
        """
        return self._com('*RST')

    def operation_complete(self):
        """
        查询操作是否完成

        返回:
            int: 1表示操作完成，0表示还在执行

        使用实例:
            >>> while not vna.operation_complete():
            ...     time.sleep(0.1)  # 等待操作完成
        """
        return self._com("*OPC?")

    def opc(self):
        """
        查询操作是否完成（operation_complete的别名）

        返回:
            int: 1表示操作完成

        使用实例:
            >>> vna.opc()
        """
        return self.operation_complete()

    def get_sweep_time(self):
        """
        查询扫描时间

        返回:
            str: 扫描时间（秒）

        使用实例:
            >>> sweep_time = float(vna.get_sweep_time())
            >>> print(f"扫描时间: {sweep_time:.3f}秒")
        """
        return self._com("SENS:SWE:TIME?")


    ########################################
    # parameters
    ########################################
    def set_trigger(self, source='bus', averaging=0, initiate="single"):
        """
        设置触发参数组合

        该方法同时设置触发源、平均状态和初始化模式。

        参数:
            source: 触发源，默认'bus'
            averaging: 平均状态，默认0(关闭)
            initiate: 初始化模式，默认'single'

        使用实例:
            >>> vna.set_trigger('bus', 0, 'single')  # 设置总线触发，无平均，单次模式
            >>> vna.set_trigger('internal', 1, 'cont')  # 内部触发，开启平均，连续模式
        """
        self.trigger_source(source)
        self.trigger_averaging(averaging)
        self.trigger_initiate(initiate)

    def set_averaging(self, state=None, count=0):
        """
        设置平均参数组合

        该方法同时设置平均状态和平均次数。

        参数:
            state: 平均状态，None表示不改变
            count: 平均次数，默认0

        使用实例:
            >>> vna.set_averaging('on', 10)   # 开启平均，设置10次平均
            >>> vna.set_averaging('off', 0)   # 关闭平均
        """
        state = str(state)
        self.average_state(state)
        self.average_count(count)

    def set_freq_axis(self, start="", stop="", center="", span="", point=1000, bandwidth=1000, sweep_type='lin'):
        """
        设置频率轴参数组合

        该方法同时设置频率范围、测量点数、带宽和扫描类型。

        参数:
            start: 起始频率（Hz），空字符串表示不设置
            stop: 终止频率（Hz），空字符串表示不设置
            center: 中心频率（Hz），空字符串表示不设置
            span: 频率范围（Hz），空字符串表示不设置
            point: 测量点数，默认1000
            bandwidth: 中频带宽（Hz），默认1000
            sweep_type: 扫描类型，默认'lin'

        使用实例:
            >>> vna.set_freq_axis(1e9, 2e9, point=1001)  # 设置1-2GHz，1001点
            >>> vna.set_freq_axis(center=1.5e9, span=1e9)  # 中心1.5GHz，范围1GHz
        """
        if type(start) != str:
            self.freq_start(start)
        if type(stop) != str:
            self.freq_stop(stop)
        if type(center) != str:
            self.freq_center(center)
        if type(span) != str:
            self.freq_span(span)

        self.points(point)
        self.bandwidth(bandwidth)
        self.sweep_type(sweep_type)

    def set_response_axes(self, trace_formats, delay, phase_offset, s_par='S12'):
        """
        设置响应轴参数组合

        该方法设置多条迹近的显示格式、延迟和相位偏移。

        参数:
            trace_formats: 迹近格式列表或单个格式字符串
            delay: 电气延迟时间
            phase_offset: 相位偏移值
            s_par: S参数类型，默认'S12'

        使用实例:
            >>> vna.set_response_axes(['mlog', 'phase'], 1e-9, 90, 'S21')
            >>> # 设置两条迹近：对数幅度和相位，延迟1ns，相位偏移90度
        """
        if type(trace_formats) == str:
            trace_formats = [trace_formats]
        if s_par == str:
            s_par = [s_par] * len(trace_formats)
        self.delay(delay)
        self.phase_offset(phase_offset)
        self.traces_number(len(trace_formats))
        for i, trace_format in enumerate(trace_formats):
            self.active_trace(i + 1)
            self.s_par(s_par)
            self.format_trace(trace_format)

    def get_parameters(self, chan=""):
        """
        获取当前仪器的所有参数设置

        该方法返回一个字典，包含所有重要的仪器参数。

        参数:
            chan: 通道编号

        返回:
            dict: 包含各种参数的字典

        使用实例:
            >>> params = vna.get_parameters()
            >>> print(f"起始频率: {params['freq_start']} Hz")
            >>> print(f"终止频率: {params['freq_stop']} Hz")
        """

        # total_traces = self.traces_number(chan)

        parameters = {'freq_start': self.freq_start(),
                      'freq_stop': self.freq_stop(),
                      'freq_center': self.freq_center(),
                      'freq_span': self.freq_span(),
                      'points': self.points(),
                      'bandwidth': self.bandwidth(),
                      'format_trace': self.format_trace(),  # get this for each channel
                      's_par': self.s_par(),  # get this for each channel
                      'power': self.power,
                      'average_count': self.average_count(),
                      'average_state': self.average_state(),
                      'delay': self.delay(),
                      'phase_offset': self.phase_offset()
                      }
        return parameters

    def set_parameters(self, chan="", **kwargs):
        """
        批量设置仪器参数

        该方法允许通过关键字参数批量设置仪器参数。

        参数:
            chan: 通道编号
            **kwargs: 参数名和值的对应关系

        使用实例:
            >>> vna.set_parameters(freq_start=1e9, freq_stop=2e9, points=1001)
            >>> vna.set_parameters(average_state='on', average_count=10)
        """
        # TODO: test this and read get **kwargs from a config file
        parameters = {'freq_start': self.freq_start,
                      'freq_stop': self.freq_stop,
                      'freq_center': self.freq_center,
                      'freq_span': self.freq_span,
                      'points': self.points,
                      'bandwidth': self.bandwidth,
                      'format_trace': self.format_trace,  # set this for each channel
                      's_par': self.s_par,  # set this for each channel
                      'power': self.power,
                      'average_count': self.average_count,
                      'average_state': self.average_state,
                      'delay': self.delay,
                      'phase_offset': self.phase_offset
                      }

        # need to figure out how to format the trace specific parameters like format
        for i in kwargs.keys():
            try:
                parameters[i](kwargs[i])
            except KeyError:
                pass

    ##############################
    # send commands
    ##############################

    def _com(self, cmd):
        """
        内部通信方法，用于发送SCPI命令

        该方法处理SCPI命令的发送和接收，自动区分查询命令和设置命令。

        参数:
            cmd: SCPI命令字符串

        返回:
            查询命令（以'?'结尾）返回数值或字符串；设置命令返回确认信息
        """
        if self.verbatim:
            print(cmd)
        if cmd[-1] == '?':
            value = self._inst.query(cmd)
            try:
                return float(value)
            except:
                return value
            # try:
            #     return float(value)
            # except:
            #     return value
        else:
            self._inst.write(cmd)
            return "Sent: " + cmd

    def _com_binary(self, cmd):
        """
        内部二进制通信方法，用于高速数据传输

        该方法优化了大量数据的传输速度，使用二进制格式而非 ASCII。

        参数:
            cmd: SCPI命令字符串

        返回:
            查询命令返回 numpy 数组；设置命令返回确认信息
        """
        if self.verbatim:
            print(cmd)
        if cmd[-1] == '?':
            return self._inst.query_binary_values(cmd, datatype='d', is_big_endian=True)
        else:
            # TODO: Test this section
            self._inst.write_binary_values(cmd, datatype='d', is_big_endian=True)
            return "Waveform sent"

def main():
        ################
    # Create object/Connect to device.
    ################
    rm = visa.ResourceManager('@py')
    ip = '192.168.0.100'
    vna = E5071C("TCPIP::{}::INSTR".format(ip))
    ################
    # Set up parameters related to frequency scan.
    ################
    vna.freq_start(6.17)
    vna.freq_stop(6.25)
    vna.points(1001)
    vna.bandwidth(1000)
    vna.sweep_type('lin')
    ################
    # Set up trace related commands. Channel related commands are similar.
    ################
    vna.traces_number(2)
    vna.active_trace(1)
    vna.s_par('S33')
    print(vna.format_trace('mlog'))
    vna.delay(1)
    vna.phase_offset(15)
    print(vna.active_trace(2))
    vna.s_par('S33')
    vna.delay(1)
    vna.phase_offset(15)
    print(vna.format_trace('phase'))
    # print(vna.active_trace(3))
    # vna.s_par('S12')
    # vna.delay(1)
    # vna.phase_offset(180)
    # print(vna.format_trace('Plog'))
    ################
    # Set up averaging parameters. Don't forget to set the "vna.trigger_averaging(True)" when using averaging
    ################
    print(vna.average_state(False))
    print(vna.average_count(0))
    ################
    # Set up averaging parameters.
    ################
    print(vna.trigger_source('bus'))
    print(vna.trigger_averaging(0))
    print(vna.trigger_initiate('single'))
    print(vna.trigger_now())
    ################
    # Read the data on the screen
    ################
    # print(vna.read_freq())
    # print(vna.read_trace(1)[0])
    # print(vna.read_trace(2)[0])
    # data = vna.read_trace(3)
    # print(data[0])
    # print(data[1])
    data = vna.read_all_traces()  # This command gets values for x axis and the primary and secondary data for all the traces.

if __name__ == "__main__":
    """
    Keysight E5071C VNA 使用示例

    本示例演示了如何使用该驱动程序连接并控制 Keysight E5071C 矢量网络分析仪。
    包括连接设备、设置测量参数、配置迹近、读取数据等操作。
    """
    ################
    # 创建对象/连接到设备
    # Create object/Connect to device.
    ################
    rm = visa.ResourceManager('@py')
    ip = '192.168.0.100'  # 替换为实际的VNA IP地址
    vna = E5071C("TCPIP::{}::INSTR".format(ip))

    print("连接成功！设备信息:", vna.identify())

    ################
    # 设置频率扫描相关参数
    # Set up parameters related to frequency scan.
    ################
    print("\n=== 设置频率参数 ===")
    vna.freq_start(6.17e9)    # 起始频率 6.17 GHz
    vna.freq_stop(6.25e9)     # 终止频率 6.25 GHz
    vna.points(1001)          # 1001个测量点
    vna.bandwidth(1000)       # 中频带宽 1 kHz
    vna.sweep_type('lin')     # 线性扫描
    print(f"频率范围: {vna.freq_start():.2e} - {vna.freq_stop():.2e} Hz")
    print(f"测量点数: {vna.points()}")

    ################
    # 设置迹近相关命令。通道相关命令类似。
    # Set up trace related commands. Channel related commands are similar.
    ################
    print("\n=== 配置迹近 ===")
    vna.traces_number(2)      # 设置2条迹近

    # 配置第1条迹近：S33参数，对数幅度显示
    vna.active_trace(1)
    vna.s_par('S33')
    print("第1条迹近:", vna.format_trace('mlog'))
    vna.delay(1e-9)           # 设置延迟 1ns
    vna.phase_offset(15)      # 设置相位偏移 15度

    # 配置第2条迹近：S33参数，相位显示
    print("切换到第2条迹近:", vna.active_trace(2))
    vna.s_par('S33')
    vna.delay(1e-9)
    vna.phase_offset(15)
    print("第2条迹近:", vna.format_trace('phase'))

    ################
    # 设置平均参数。使用平均时别忘设置"vna.trigger_averaging(True)"
    # Set up averaging parameters. Don't forget to set "vna.trigger_averaging(True)" when using averaging
    ################
    print("\n=== 设置平均参数 ===")
    print("平均状态:", vna.average_state('off'))  # 关闭平均
    print("平均次数:", vna.average_count(0))     # 设置为0

    ################
    # 设置触发参数
    # Set up trigger parameters.
    ################
    print("\n=== 设置触发参数 ===")
    print("触发源:", vna.trigger_source('bus'))        # 设置为总线触发
    print("触发平均:", vna.trigger_averaging('off'))   # 关闭触发平均
    print("触发模式:", vna.trigger_initiate('single')) # 设置为单次触发

    ################
    # 执行测量并等待完成
    # Execute measurement and wait for completion
    ################
    print("\n=== 执行测量 ===")
    print(触发测量结果:", vna.trigger_now())

    ################
    # 读取屏幕上的数据
    # Read the data on the screen
    ################
    print("\n=== 读取数据 ===")

    # 读取频率轴
    freq_axis = vna.read_freq()
    print(f"频率轴数据点数: {len(freq_axis)}")
    print(f"频率范围: {freq_axis[0]/1e9:.3f} - {freq_axis[-1]/1e9:.3f} GHz")

    # 读取单条迹近数据
    real_part, imag_part = vna.read_trace(1)
    print(f"第1条迹近数据点数: {len(real_part)} (实部), {len(imag_part)} (虚部)")

    # 计算幅度和相位
    magnitude_db = 20 * np.log10(np.sqrt(real_part**2 + imag_part**2))
    phase_deg = np.arctan2(imag_part, real_part) * 180 / np.pi
    print(f"幅度范围: {magnitude_db.min():.2f} - {magnitude_db.max():.2f} dB")
    print(f"相位范围: {phase_deg.min():.2f} - {phase_deg.max():.2f} 度")

    # 读取所有迹近数据
    all_data = vna.read_all_traces()  # 这个命令获取x轴和所有迹近的主数据和辅助数据
    print(f"\n所有数据形状: {all_data.shape}")
    print("数据结构:")
    print("  第0行: 频率轴")
    print("  第1行: 迹近1实部")
    print("  第2行: 迹近1虚部")
    print("  第3行: 迹近2实部")
    print("  第4行: 迹近2虚部")

    ################
    # 示例: 使用便捷方法设置参数
    # Example: Using convenience methods to set parameters
    ################
    print("\n=== 便捷方法示例 ===")

    # 一次性设置频率参数
    vna.set_freq_axis(start=5e9, stop=7e9, point=501, bandwidth=1000)
    print("已设置频率范围: 5-7 GHz, 501点")

    # 一次性设置触发参数
    vna.set_trigger(source='internal', averaging=0, initiate='cont')
    print("已设置内部触发，连续模式")

    # 一次性设置平均参数
    vna.set_averaging('on', 5)
    print("已开启5次平均")

    ################
    # 获取当前参数设置
    # Get current parameter settings
    ################
    print("\n=== 当前参数设置 ===")
    params = vna.get_parameters()
    for key, value in params.items():
        if 'freq' in key:
            print(f"{key}: {value/1e9:.3f} GHz" if isinstance(value, (int, float)) else f"{key}: {value}")
        else:
            print(f"{key}: {value}")

    ################
    # 关闭连接
    # Close connection
    ################
    print("\n=== 关闭连接 ===")
    vna.close()
    print("已关闭与VNA的连接")
