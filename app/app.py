import streamlit as st

# Title of the app
st.title("My First Streamlit App")

# Adding a header and some text
st.header("Hello, Streamlit!")
st.write("This is a simple Streamlit application.")

# Create an interactive widget (slider)
age = st.slider("Select your age", 0, 100, 25)
st.write("Your selected age is:", age)

name = st.text_input("Enter your name", "Type here...")
st.write("Hello,", name)

if st.checkbox("Show text"):
    st.write("Hello, Streamlit!")

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

fig, ax = plt.subplots()
ax.plot(x, y)
st.pyplot(fig)

