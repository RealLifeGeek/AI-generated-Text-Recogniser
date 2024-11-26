AI generated text recognizer
An AI-generated text recognizer specifically tailored for short-form content, such as tweets and Facebook posts. The ultimate goal is to design a system that evaluates the likelihood of a given text being human-authored or AI-generated. 
 
Background 
As AI-generated text becomes more realistic and prevalent, distinguishing between a content created by AI or by humans is increasingly challenging. For example, misinformation campaigns, spam, or unethical use of AI can exploit the indistinguishability of such texts. 
 
This problem is rapidly becoming widespread as AI-generated text tools (like ChatGPT, Gemini, or other generative AI platforms) gain popularity and accessibility. Social media platforms, blogs, and forums now frequently feature AI-generated posts, with many users unknowingly consuming or interacting with such content. This is particularly significant in domains like politics, advertising, and journalism, where authenticity is critical. Detecting AI-authored content could play a crucial role in maintaining transparency and trust. 
 
The recognizer could be used for educational purposes and coudl be integrated into educational platforms to teach others about AI's role and influence on communication. 
 
With growing concerns about AI misuse, such as deepfakes or automated misinformation, this project is timely and could contribute to better moderation tools for social platforms. 
 
Data and AI techniques 
The project involves creating a labeled database of AI- and human-generated posts and employing a machine learning approach to classify new inputs. The initial method focuses on TF-IDF-based text representation combined with nearest-neighbor analysis to identify stylistic and linguistic similarities. 
 
As an AI-generated content data source could be used texts generated by popular AI models such as: 
OpenAI's ChatGPT (including different versions like GPT-3.5 and GPT-4) 
Google's Bard 
Meta's Llama 
Other AI writing tools like Jasper, Writesonic, or Copy.ai 
 
For a human-created contend there is a plenty of options like: 
Publicly available social media platforms like Twitter and Facebook. 
Publicly available corpora of conversaional or informal text (e.g., Reddit comments or Kaggle datasets of tweets). 
User-generated content repositories, such as open forums or archives of short-form writing. 
 
How is it used 
The AI-generated text recognizer is designed to operate primarily in the context of social media platforms like Twitter and Facebook, where short-form text dominates. It can be used as a standalone tool or integrated into other systems (e.g., moderation tools, content analysis platforms). 
 
Challenges 
Edited or hybrid content: If an AI-generated post is slightly edited by a human, or vice versa, your system may struggle to accurately classify it. 
 
Context understanding: The tool doesn’t consider the broader context of the post, such as who authored it, their writing style, or the cultural and situational nuances. 
 
Misuse of detection: The recognizer identifies patterns but doesn't explain why a post was classified as AI-generated. This could lead to ethical concerns if the tool is used to accuse individuals of dishonesty or to discredit genuine posts. 
 
The neural network in the provided code isn't trained! It initializes the weights randomly, which means the model has no learned knowledge about how to classify human vs. AI text. The model is merely passing random values through the network and outputting probabilities based on those initial random weights. 
 
What next 
Expanding the recognizer to include embeddings can significantly improve its ability to understand the context and relationships between words in tweets and posts. This would enable the detection of more subtle differences between AI and human writing styles, even in cases of sophisticated editing or hybrid content. 
 
Acknowledgments 
The chapter Working with text from AI building course has played a crucial role in this project. I have been inspired especially by Exercise 18: TF – IDF. Since I am aware that this method has some challenges, it could greatly work as a starting point for broader application. 
