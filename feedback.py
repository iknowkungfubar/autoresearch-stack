class Feedback:
    def reward(self,val_bpb,score):
        return (1/(val_bpb+1e-6)) + score/100

    def log(self,step,val_bpb,score):
        r=self.reward(val_bpb,score)
        print("R:",step,val_bpb,score,r)
        return r
