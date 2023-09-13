from flask import Flask, render_template, request
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")

app=Flask(__name__,template_folder='template')

# Load the model from the pickle file
classification_model = pickle.load(open('Models/Classification.pkl','rb'))
regression_model = pickle.load(open('Models/Regression_num.pkl','rb'))


def prediction(model , inputs):
    # Perform prediction using the loaded model
    return model.predict(inputs)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    
    if request.method == 'POST':
        # //Classification
        eligibility = "NOT ELIGIBLE"
        pred  = ["--","--","--"]
        emi = pred[0]
        roi = pred[1]
        eligible_loan_amount = pred[2]
        bids_portfolio_manager = request.form.get('bidsPortfolioManager')
        interest = request.form.get('interest')
        monthly_payment = request.form.get('monthlyPayment')
        rating = request.form.get('rating')
        prev_loans = request.form.get('noOfPreviousLoansBeforeLoan')
        principal_payments_made = request.form.get('principalPaymentsMade')
        principal_balance = request.form.get('principalBalance')
        interest_penalty_balance = request.form.get('interestAndPenaltyBalance')
        prev_loans_amount = request.form.get('amountOfPreviousLoansBeforeLoan')
        prev_repayments = request.form.get('previousRepaymentsBeforeLoan')

        # regression
        age = request.form.get('age')
        appliedAmount = request.form.get('appliedAmount')
        debtToIncome = request.form.get('debtToIncome')
        loanTenure = request.form.get('loanTenure')
        useOfLoan = request.form.get('useOfLoan')
        incomeTotal = request.form.get('incomeTotal')
        gender = request.form.get('gender')
        maritalStatus = request.form.get('maritalStatus')
        interestAmount = request.form.get('interestAmount')
        occupationArea = request.form.get('occupationArea')
        
        inputs_classifying = [[
            bids_portfolio_manager,
            interest,
            monthly_payment,
            rating,
            prev_loans,
            principal_payments_made,
            principal_balance,
            interest_penalty_balance,
            prev_loans_amount,
            prev_repayments
        ]]
        
        inputs_regression = [[
            age,
            appliedAmount,
            debtToIncome,
            loanTenure,
            useOfLoan,
            incomeTotal,
            gender,
            maritalStatus,
            interestAmount,
            occupationArea            
        ]]
        inputs_regression = np.array(inputs_regression, dtype=np.float32)
        pred = regression_model.predict(inputs_regression)
        classified = classification_model.predict(inputs_classifying)
        
        # Extract the prediction values
        if(classified[0]):
            eligibility = "ELIGIBLE"
            emi = round(pred[0][0], 3)
            eligible_loan_amount = round(pred[0][1], 3)
            roi = round(pred[0][2], 3)

        return render_template('index.html', eligibility=eligibility, emi=emi, roi=roi, eligible_loan_amount=eligible_loan_amount)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
