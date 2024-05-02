# Overview
This project uses machine learning to predict customer lifetime value in vehicle insurance. Evaluating the performance of 3 different forecasting models - random forest, gradient boosting, and polynomial to select the best one. Additionally, I considered the features that influenced significantly the model and selected the right features for the model.
# About Dataset
<a href = "https://www.kaggle.com/datasets/ranja7/vehicle-insurance-customer-data">Dataset</a> is customer data with their car insurance policies. The detailed information about customers and their vehicle insurance coverage provided can be explored to segment similar customer types.

| No | Column Name | Note | 
| --- | --- | --- |
| 1 | Customer |  Mã khách hàng |
| 2 | State | Khu vực của khách hàng |
| 3 | Response |  Trạng thái phản hồi của khách hàng (có hoặc không) |
| 4 | Coverage |  Loại bảo hiểm (Basic, Extended, Premium) |
| 5 | Education |Trình độ học vấn của khách hàng  |
| 6 | Effective To Date |  Ngày bắt đầu áp dụng bảo hiểm |
| 7 | EmploymentStatus | Tình trạng việc làm của khách hàng |
| 8 | Gender | Giới tính của khách hàng |
| 9 | Income | Thu nhập của khách hàng |
| 10 | Location Code | Mã vùng địa lý của khách hàng |
| 11 | Marital Status | Tình trạng hôn nhân của khách hàng |
| 12 | Monthly Premium Auto | Số tiền phải trả hàng tháng cho bảo hiểm ô tô |
| 13 | Months Since Last Claim |  Số tháng kể từ khi khách hàng gửi yêu cầu bồi thường cuối cùng |
| 14 | Months Since Policy Inception |  Số tháng kể từ khi khách hàng mua bảo hiểm |
| 15 | Number of Open Complaints | Số lượng khiếu nại chưa giải quyết |
| 16 | Number of Policies |  Số lượng HĐ mà khách hàng đang sử dụng |
| 17 | Policy Type | Loại HĐ bảo hiểm (Personal Auto, Corporate Auto, Homeowners) |
| 18 | Policy | Mã số chính sách bảo hiểm |
| 19 | Renew Offer Type |  Loại gói bảo hiểm được cung cấp cho khách hàng khi hết hạn gói cũ (Offer1, Offer2, Offer3, Offer4) |
| 20 | Sales Channel | Kênh bán hàng (Agent, Branch, Call Center, Web) |
| 21 | Total Claim Amount | Tổng số tiền yêu cầu bồi thường trong thời gian sử dụng dịch vụ bảo hiểm |
| 22 | Vehicle Class | Loại xe (Four-Door Car, Luxury Car, SUV, Sports Car, Two-Door Car) |
| 23 | Vehicle Size |  Kích thước xe (Large, Medium, Small) |
| 24 | Customer Lifetime Value |  Giá trị khách hàng trong suốt thời gian sử dụng dịch vụ bảo hiểm |
# Process
## 1. Analyzing distribution of features.
### Customer Lifetime Value
  
![image](https://github.com/LanLTH/Project-3--CLV-in-Vehicle-Insurance/assets/160514942/3ce6a07a-7f64-4ba6-be61-cd7dda102e86)

CLV distribution is skewed to the right, most values are in the range of 2000-10000
### Distribution of Numerical Group Variable
  
![image](https://github.com/LanLTH/Project-3--CLV-in-Vehicle-Insurance/assets/160514942/c87c78a9-a8a9-4b8f-b5c4-f0805b60f8a9)
Except for the two variables: Monthly Premium Auto and Total Claim Amount, the remaining variables have no outliers.
Because there will be customers who are eligible and will sign up for insurance contracts with higher benefits and limits, corresponding to high insurance fees. Accordingly, the compensation amount is also higher
### Analyzing the relationship between numerical group variables
  
  ![image](https://github.com/LanLTH/Project-3--CLV-in-Vehicle-Insurance/assets/160514942/12e78455-38a5-463c-ac68-6f8f526127a4)

**If data points tend to increase or decrease together, a correlation between the two variables can be demonstrated. If the data points are randomly scattered, it may indicate that there is no clear relationship between the two variables.**<br>

* The histograms on the diagonal represent the distribution of each variable. It can be seen that most of the data does not follow a normal distribution and is skewed to the right.
* CLV& Premium or Premium &Total claim has a V-shaped distribution, concentrated at the origin of the coordinates increasing and widely dispersed, showing a non-linear relationship between the two variables
* The data points in the graph of number of complaints, number of policies vs the remaining variables are distributed on parallel horizontal lines but the correlation coefficient is very small (about 0.001), this shows that there is no relationship. non-linearity between "Number of Open Complaints" and other variables.
* The remaining graphs are randomly distributed and do NOT have a linear relationship with each other. 

### Analyze the distribution of variables in non-numerical groups
![image](https://github.com/LanLTH/Project-3--CLV-in-Vehicle-Insurance/assets/160514942/7e85dc38-ea7d-4675-a380-3db59b796fc1)
## 2. Build and evaluate models
###  Polynomial Model
![image](https://github.com/LanLTH/Project-3--CLV-in-Vehicle-Insurance/assets/160514942/0f4134a7-72c6-46a4-b9c4-3f9c2da1e318)
![image](https://github.com/LanLTH/Project-3--CLV-in-Vehicle-Insurance/assets/160514942/dbfa7d54-c105-49e2-86f4-c322f73724e8)
* The red line represents forecast values and the blue line represents observed values. 
* The forecast line results from the poly model are deviating quite a lot from the actual value. 
* Even though the RSE, RMSE, and MAE are good, the R2_score is very low, less than 20%, meaning the model is only able to explain 20% of the dependence of the variables on the target variable.
==> the model is not suitable for this data set
###  Random Forest

![image](https://github.com/LanLTH/Project-3--CLV-in-Vehicle-Insurance/assets/160514942/4f0990f8-7a5e-4851-a2ae-3b71d3a08056)

![image](https://github.com/LanLTH/Project-3--CLV-in-Vehicle-Insurance/assets/160514942/2e82de85-dc53-4ca8-9129-3bd7a84337bf)

* The results obtained from the RF model, although not very high, are much better than the poly model. 
* It can be seen that the predicted value line and the observed value are relatively similar, with a slight deviation in this part. 
* The model is capable of explaining about 70% of the dependence of the variables on the target variable

### Gradient Boosting

![image](https://github.com/LanLTH/Project-3--CLV-in-Vehicle-Insurance/assets/160514942/4f7e50e6-bfe3-4060-b569-c4ff1eef8b79)
![image](https://github.com/LanLTH/Project-3--CLV-in-Vehicle-Insurance/assets/160514942/fe6485a0-306a-4add-a5c4-22bc0d2c8c24)

The obtained r2_score result is equal to the RF model, however based on the chart it can be seen that the GB model's forecast line is more different than the RF model's forecast line.

### Evaluate the impact of Features
![image](https://github.com/LanLTH/Project-3--CLV-in-Vehicle-Insurance/assets/160514942/6ce6198a-3115-46c7-bcc3-64e9d04223d3)
Variables have a high degree of influence on forecasting CLV là Income , Monthly Premium Auto, Months Since Last Claim, Months Since Policy Inception, Total Claim Amount, Number of Policies
