#random search class
#not used in the project
import random
class RandomBitSearch:
    def __init__(self,qubo_utility):
        self.qubo_utility=qubo_utility
    
    def prepare_random_bit(self):
        bit_list=random.choices([0,1],k=self.qubo_utility.bit_length)
        bit_list=self.qubo_utility.calc_interactions(bit_list)
        return bit_list
        
    def explore_batch(self,model,batch_size=3200):
        explore_list=[self.prepare_random_bit() for i in range(batch_size)]
        predY=model.predict(explore_list)
        
        res_list=list(zip(predY,explore_list))
        res_list=sorted(res_list,reverse=True)
        
        best_res=res_list[0]
        return best_res[0],self.qubo_utility.get_uninteracted_bit(best_res[1])