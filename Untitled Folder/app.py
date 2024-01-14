import pickle
import streamlit as st

model = pickle.load(open("spam2.pkl", "rb"))
cv = pickle.load(open("vectorizer.pkl", "rb"))

def main():
    st.title("Email Spam Detector Apps")
    st.header('Built with Streamlit and Python')
    
    msg = st.text_input("Enter the text: 🤞🤞")

    if st.button("Predict"):
        df = [msg]
        vect = cv.transform(df).toarray()
        prediction = model.predict(vect)
        result = prediction[0]
        
        if result == 1:
            st.error("This is a spam mail")
        else:
            st.success("This is not a spam mail")

if __name__ == "__main__":
    main()
