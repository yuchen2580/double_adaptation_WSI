

# General BoVW Pipeline For Training and Testing.

The notebook nb_sampleusage_bovwpipeline trains and saves the model in the folder Trained_Models.
The notebook collect_stats firstly loads the trained model, and then collects some statistics from the whole dataset. 


Parts that may require modification:
* In the collect_stats notebook, there is a part as follows:
```
dl_of_statcollector = DLWithInTurnSched(
                              dataset=val_ds,\
                                ......
``` 
dataset is the dataset on which the evaluation is performed. You may need to change that to, e.g., testing dataset or the whole dataset.

* In the collect_stats notebook, the following function specifies when collecting has to be sopped:
```
@abstractmethod
    def get_flag_finishcollecting(self):
        self.num_calls_to_getflagfinished += 1
        print("self.num_calls_to_getflagfinished = {}\n"\
                  .format(self.num_calls_to_getflagfinished))
        list_statcount = []
        for patient in self.dict_patient_to_accumstat.keys():
            if(self.dict_patient_to_accumstat[patient] != None):
                list_statcount.append(self.dict_patient_to_accumstat[patient]["count"])
            else:
                list_statcount.append(0)
        print(" numstats in [{} , {}],     num zeros = {}"
              .format(min(list_statcount) , max(list_statcount),
                      np.sum(np.array(list_statcount) == 0)) )
        
        if(min(list_statcount) > 0):
            if(self.num_calls_to_getflagfinished > 100):
                return True
        else:
            return False
```
In the above code, `num_zeros` is the number of patients with 0 stats.
The part `if(min(list_statcount) > 0):` means: for all patients, some stats has to be collected, otherwise the return value is false.
The field `self.num_calls_to_getflagfinished` is the number of calls to the above functions. This function is called every 5 seconds.
As a rule of thumb, 100 in the above code has to be set such that for 250 WSIs, the statcollection will be running for about 12 hours. Meaning that for 500 WSIs, about 24 hours, ect. 



