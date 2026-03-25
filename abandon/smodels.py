class SpecialSM(SurrogateModel):
    def __init__(self):
        model_ge = HFSSNet(  # Pattern -> Response
            AntennaPattern.getAllPixel(), AntennaResponse.size()
        )
        criterion_ge = nn.MSELoss()
        optimizer_ge = Ranger(params=model_ge.parameters(), lr=config.lr)
        self.scheduler_ge = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_ge, mode="min", factor=0.5, patience=10, min_lr=1e-6
        )
        super().__init__(model_ge, criterion_ge, optimizer_ge)

    def train(self, pattern):
        self.model.train()
        pattern = tensor(pattern)
        sm_loss = []
        for epoch_ge in range(500):
            self.progress_callback(epoch_ge, 500)
            self.optimizer.zero_grad()

            response = self.model(pattern)

            match pattern.size(0):
                case 625:  # ? 25*25
                    s11 = AntennaResponse(response[0])
                    s21 = AntennaResponse(response[1])
                    s22 = AntennaResponse(response[2])

                    loss_s11 = s11.criterion("S11")
                    loss_s21 = s21.criterion("S21")
                    loss_s22 = s22.criterion("S22")

                    loss_ge: Tensor = loss_s11 + loss_s21 + loss_s22
                case 1600:  # ? 40*40
                    loss_ge: Tensor = AntennaResponse(response).criterion()
                case _:
                    raise ValueError(f"No matching settings found for {pattern.size(0)}")

            loss_ge.backward()
            self.optimizer.step()
            self.scheduler_ge.step(loss_ge.item())
            sm_loss.append(loss_ge.item())

        return sm_loss
