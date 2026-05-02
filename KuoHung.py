from antenna import *
config.device = "cpu"

from antenna.utils import *
from antenna.utils.data import Data

#* Select according to actual application.
from antenna.patch import SinglePortSimulator, DualPortSimulator
# AntennaPattern.setDefaultCoordinate((0, 200, 0, 1))

class KuoHung:
    def __init__(self, name:str, port:Literal['Single', 'Dual']):
        #* Basic Config
        connect_network_drive("T:", r"\140.123.106.219\temp", "user", "ailab120")
        AntennaPattern.setDefaultCoordinate((0, 25, 0, 25))
        # PATTERN_SIZE = AntennaPattern.size(flatten=True)
        # RESPONSE_SIZE = AntennaResponse.size(flatten=True)

        self.name = f"KuoHung-{name}"
        self.result_path = DATASET_PATH.joinpath('KuoHung Pattern')
        
        self.data = Data(name=self.name, rootdir=self.result_path)
        
        match port:
            case 'Single':
                self.simulator = SinglePortSimulator(self.result_path)
                self.nrowcol = (1, 3)
                self.label = ('S11', 'Gain')

                AntennaResponse.registerLabels(*self.label, x = 'n257')

            case 'Dual':
                self.simulator = DualPortSimulator(self.result_path)
                self.nrowcol = (2, 2)
                self.label = ('S11', 'S21', 'S22')

                AntennaResponse.registerLabels(*self.label, x = 'n257')

            case _:
                raise ValueError(f"{port}")

        AntennaPattern.register_simulator(self.simulator)
        self.x = AntennaResponse.x()
    
    def __call__(self):
        if self.data.savepath.exists():
            self.draw()
        else:
            self.simulate()
            self.draw()
    
    @staticmethod
    def load(name:str) -> tuple[Tensor, Tensor]:
        return Data(name=f"KuoHung-{name}", rootdir=DATASET_PATH.joinpath('KuoHung Pattern')).load()
    
    def __str__(self):
        return f"<KuoHung name={self.name} file={self.data.savepath}>"

    def simulate(self, pattern:Optional[AntennaPattern] = None):
        pattern = pattern or self.pattern
        self.simulator.open()
        self.simulator.start(self.name)
        response = pattern.simulate()
        self.simulator.end()

        data_in = Data([~pattern, ~response], name=self.name, rootdir=self.result_path)
        data_in.save()
    
    def draw(self):
        data_result = Data(name=self.name, rootdir=self.result_path)
        KuoHung, responses = data_result.load()

        with Figure(self.name, self.nrowcol, save=True, rootdir=self.result_path, size=(18, 9)) as fig:
            fig.addAll()

            AntennaPattern(KuoHung).plot(fig[0])

            for n, (label, response) in enumerate(zip(self.label, responses), start=1):

                fig[n].plot(self.x, response)
                fig[n].set_title(label)
    
    def single_1(self):
        #* 13 * (0, 12, 1, 12, 0)
        upper_part = torch.zeros(12, 25)
        lower_part = torch.cat((torch.zeros(13, 1), torch.ones(13, 23), torch.zeros(13, 1)), dim=1)
        self.pattern = AntennaPattern(
            torch.cat((upper_part, lower_part), dim=0)
        )


    def single_2(self):
        #* 13 * (0, 12, 1, 12, 0)
        upper_part = torch.zeros(12, 25)
        lower_part = torch.ones(13, 25)
        self.pattern = AntennaPattern(
            torch.cat((upper_part, lower_part), dim=0)
        )

if __name__ == "__main__":
    base1 = KuoHung('1', port='Single')
    base2 = KuoHung('2', port='Single')

    pattern1, responses1 = base1.data.load()
    pattern2, responses2 = base2.data.load()

    
    with Figure('Base', (2,3), show=True, size=(18, 9), default_axes_title_size=16, default_tick_size=14) as fig:
            
            # #* Base 1
            # title1 = f"Base-1"

            # pattern1_ax = fig.index(-1)
            # AntennaPattern(pattern1).plot(pattern1_ax)
            # pattern1_ax.set_title(title1)

            # s1_ax = fig.index(-1)
            # s1_ax.plot(base1.x, responses1[0])
            # s1_ax.set_title(f"S11 ({title1})")

            # gain1_ax = fig.index(-1)
            # gain1_ax.plot(base1.x, responses1[1])
            # gain1_ax.set_title(f"Gain ({title1})")
            
            # #* Base 2
            # title2 = f"Base-2"

            # pattern2_ax = fig.index(-1)
            # AntennaPattern(pattern2).plot(pattern2_ax)
            # pattern2_ax.set_title(title2)

            # s2_ax = fig.index(-1)
            # s2_ax.plot(base1.x, responses2[0])
            # s2_ax.set_title(f"S11 ({title2})")

            # gain2_ax = fig.index(-1)
            # gain2_ax.plot(base2.x, responses2[1])
            # gain2_ax.set_title(f"Gain ({title2})")
            
            fig.addAll()

            for n, (base, responses) in enumerate([base1.data.load(), base2.data.load()], 0):
                title = f"Base-{n+1}"
                AntennaPattern(base).plot(fig[3*n])
                fig[3*n].set_title(title)


                fig[3*n+1].plot(base1.x, responses[0])
                fig[3*n+1].set_title(f"S11 ({title})")
                fig[3*n+1].set_xlabel("Frequency (GHz)")
                fig[3*n+1].set_ylabel("dB")

                fig[3*n+2].plot(base1.x, responses[1])
                fig[3*n+2].set_title(f"Gain ({title})")
                fig[3*n+2].set_xlabel("Frequency (GHz)")
                fig[3*n+2].set_ylabel("dB")

