#import necessary libraries
from faker import Faker
import pandas as pd
import numpy as np
import time

fake = Faker()

#create a function for creating realistic customer data, including name, DOB, ID etc.
def create_data(x):
	person ={}
	for i in range(0,x):
		person[i] = {}
		person[i]['Name'] = fake.name()
		person[i]['DOB'] = fake.date_of_birth(minimum_age=16,maximum_age=120)
		#person[i]['ID'] = np.random.choice(np.arange(1000000,9999999),replace=False) #this will avoid duplicate ID numbers but runs slowly, test on your system with a small value before generating large datasets.
		person[i]['ID'] = np.random.randint(1000000,9999999) #this will include duplicate ID numbers, but runs very quickly. It should be used unless duplicates are undesireable.
		person[i]['Address'] = fake.address()
		person[i]['Country'] = fake.country()
	return  person

tic = time.perf_counter() #starts a timer
df = pd.DataFrame(create_data(100)).transpose() #runs def create_data(number of rows required in output goes here)
toc = time.perf_counter() #ends the timer
print(f"Synthetic data creation time was: {toc - tic:0.6f} seconds") #returns text and the amount of time the process required
print(df.head()) #prints first and last entries of the dataframe

df.to_csv("customers.csv") #outputs the dataframe to a csv file