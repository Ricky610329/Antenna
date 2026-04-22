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

    
    with Figure('Base', (2,3), show=True, size=(18, 9), default_axes_title_size=18, default_tick_size=16) as fig:
            
            fig.addAll()

            for n, (base, responses) in enumerate([base1.data.load(), base2.data.load()], 0):
                title = f"Base-{n+1}"
                AntennaPattern(base).plot(fig[3*n])
                fig[3*n].set_title(title)


                fig[3*n+1].plot(base1.x, responses[0])
                fig[3*n+1].set_title(f"S11 ({title})")

                fig[3*n+2].plot(base1.x, responses[1])
                fig[3*n+2].set_title(f"Gain ({title})")

