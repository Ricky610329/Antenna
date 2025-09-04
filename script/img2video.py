#? python -m script.img2video
from sys import path
from os.path import dirname, join
path.append(join(dirname(__file__),'..'))

from antenna import get_result_path, AntennaResponse
from antenna.utils import Figure, Path, Record, config
from matplotlib.pyplot import imread
from numpy import linspace

config.setWarning()

result_path, _  = get_result_path(1751203248)
temp = Record('temp', result_path, load=True)
print(repr(temp))
print(len(temp['patch_result_buf'][-1][1]))
loss = temp['pilotLoss'][-1]

returnloss_upper = AntennaResponse.registerTargetResponse(0, -10, (4, 2, 5, 2, 4), label="returnloss_upper")
returnloss_lower = AntennaResponse.registerTargetResponse(-2.5, -50, (3, 4, 3, 4, 3), label="returnloss_lower")
gain_upper = AntennaResponse.registerTargetResponse(-17, 0, (2, 3, 7, 3, 2), label="gain_upper")
gain_lower = AntennaResponse.registerTargetResponse(-22, -3, (4, 2, 5, 2, 4), label="gain_lower")

epochs = len(temp)
x = linspace(24, 32, 17)
with Figure("line", (2, 3), rootdir=result_path, size=(9, 6)) as fig:
    def update(frame):

        ax1 = fig.index(1)
        ax1.clear()
        ax1.set_title("Element")
        ax1.imshow(temp['patch_pattern_buf'][frame].reshape(25,25))
        # fig[0].axis('off')
        
        ax2 = fig.index(2)
        ax2.clear()
        ax2.set_title("LossCurve")
        ax2.plot(loss[:frame+1], label = 'real loss')
        ax2.plot(temp['fake_loss_record'][:frame+1], label = 'fake loss')
        ax2.set_ylim(0, 60)
        ax2.set_xlim(0, epochs)
        ax2.legend()
        
        #* Response
        ax4 = fig.index(4)
        ax4.clear()
        ax4.set_title("Response S11")
        ax4.plot(x, temp['patch_result_buf'][frame][0])
        ax4.plot(x,returnloss_upper, color='red')
        ax4.plot(x, returnloss_lower, color='red')
        ax4.set_ylim(-21, 1)

        ax5 = fig.index(5)
        ax5.clear()
        ax5.set_title("Response S21")
        ax5.plot(x, temp['patch_result_buf'][frame][1])
        ax5.plot(x,gain_upper, color='red')
        ax5.plot(x, gain_lower, color='red')
        ax5.set_ylim(-21, 1)
        
        ax6 = fig.index(6)
        ax6.clear()
        ax6.set_title("Response S22")
        ax6.plot(x, temp['patch_result_buf'][frame][2])
        ax6.plot(x,returnloss_upper, color='red')
        ax6.plot(x, returnloss_lower, color='red')
        ax6.set_ylim(-21, 1)
        
        fig.fig.tight_layout(pad=0.1)
        

        

        return fig
    fig.saveMP4(update, epochs, video_time=1, del_temp=True)