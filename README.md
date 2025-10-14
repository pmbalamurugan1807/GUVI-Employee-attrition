# GUVI-Employee-attrition

Streamlit project to show employee attrition with EDA and model training 

(Note: entire project is done in 1 python file(employeeattrition.py)

step 1: This involves processing of data(Employee-Attrition.csv) into a format thats more easier to model for predictions which are mainly binary encoding and one-hot encoding and drop columns with only 1 value(aka those that dont contribute to the prediction analysis)

step 2: Next step is staring to code for streamlit alongside modelling but starting with the sidebar with "Dataset Overview", "EDA and its Visualization", "Model Training & Evaluation", "Predict Attrition", "Predict Performance Rating" options all of which denote each page 

step 3: Starting with data overview , this page is mainly to overview the data which shows the missing values and the summary of data distribution 

<img width="500" height="281" alt="Screenshot (11)" src="https://github.com/user-attachments/assets/743d9758-ff87-44a8-98eb-667d6eb23893" />
<img width="500" height="281" alt="Screenshot (12)" src="https://github.com/user-attachments/assets/931307e1-5569-49e7-899d-7810829e5311" />
<img width="500" height="281" alt="Screenshot (13)" src="https://github.com/user-attachments/assets/239076b0-c5fe-44e2-94d9-a631ed234b90" />

step 4: For EDA and its Visualization , we show a bar graph showing the amount of people that stayed or left(attrition) , a graph for gender based attrition along with a performance rating count and a heatmap showing correlation between all the factors(feature) given to us in the data.

<img width="500" height="281" alt="Screenshot (28)" src="https://github.com/user-attachments/assets/267ed2e4-c231-4799-b47c-1667643f7004" />
<img width="500" height="281" alt="Screenshot (29)" src="https://github.com/user-attachments/assets/4830d9d4-e55d-4205-a4db-0b9704e48017" />
<img width="500" height="281" alt="Screenshot (30)" src="https://github.com/user-attachments/assets/2d7ae767-349e-491b-aba8-c92b638a183c" />



step 5:For model training and evaluation, we are training the model with data around attrition using randomforestclassifier due to its binary nature and this is done by splitting the given data into training and testing model (80% for training, 20% for testing) randomly. And this is shown in streamlit with model evaluation metrics that show accuracy, precision etc , a confusion matrix for the model and a bargraph to show how important each feature is from the data 

<img width="500" height="281" alt="Screenshot (16)" src="https://github.com/user-attachments/assets/7138e917-247b-436f-81e2-25bce9bac3a0" />
<img width="500" height="281" alt="Screenshot (17)" src="https://github.com/user-attachments/assets/3324b6f6-af70-4859-8bab-565fff37d912" />
<img width="500" height="281" alt="Screenshot (18)" src="https://github.com/user-attachments/assets/cef17560-2a82-475a-a87a-9ba2801b4e33" />

step 6:For Attrition, its the same steps as above for model training, with randomforestclassifier around attrition , but for streamlit this section is to display the probability for attrition from our input as shown below:

<img width="500" height="281" alt="Screenshot (24)" src="https://github.com/user-attachments/assets/d0ec0fc6-4a94-40a6-98d3-113ff4f6abe3" />
<img width="500" height="281" alt="Screenshot (25)" src="https://github.com/user-attachments/assets/2d013ee8-0546-4c4a-b869-bbbaee84619f" />


step 7:For Predict performance rating, for this the model training was done with randomforestregressor around performance rating, mainly because this is around multiple values and not binary like attrition and for streamlit this section is to display the predicted performance rating from our input as shown below:

<img width="500" height="281" alt="Screenshot (26)" src="https://github.com/user-attachments/assets/01289222-64b6-43bf-86d6-c3048e915e6b" />
<img width="500" height="281" alt="Screenshot (27)" src="https://github.com/user-attachments/assets/5da1bfc7-116e-42cc-b1ed-405d2ba77583" />




