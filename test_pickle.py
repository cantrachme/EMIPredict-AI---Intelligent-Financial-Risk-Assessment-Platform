import pickle

print("Loading eligibility model...")
pickle.load(open("models/emi_eligibility_model.pkl", "rb"))

print("Loading EMI model...")
pickle.load(open("models/max_emi_model.pkl", "rb"))

print("✅ BOTH MODELS LOAD SUCCESSFULLY")
