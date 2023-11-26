import streamlit as st
import pickle
import os
import os.path
from functions.plots import visualize_topics_over_time,visualize_documents,visualize_hierarchy,visualize_topics

@st.cache_resource()
def load_negative_model_data():
    # Open the pickle file in binary mode for reading
    with open('results/saved_data/negative_themes/negative_data.pkl', 'rb') as f:
        # Unpickle the dictionary from the file
        neg_metadata = pickle.load(f)
    return neg_metadata

@st.cache_resource()
def load_positive_model_data():
    with open('results/saved_data/positive_themes/positive_data.pkl', 'rb') as f:
        # Unpickle the dictionary from the file
        pos_metadata = pickle.load(f)

    return pos_metadata

def main():
    # Set the title and header of the app
    st.title("Topic Modeling")
    st.header("Visualization of Topics")
    # Get the absolute path of the current directory

    # Write the current directory path to the app interface
    neg_metadata = load_negative_model_data()
    pos_metadata = load_positive_model_data()

    tab_neg, tab_pos,tab_compare = st.tabs(["Negative Feedback", "Positive Feedback", "Comparative Analysis"])
    with tab_neg:
        # Get model and metadata from World of Tanks
        # neg_wot_model = neg_models['World of Tanks Blitz']
        neg_wot_metadata = neg_metadata['World of Tanks Blitz']
        wot_max_topics = max(neg_wot_metadata['topics_over_time']['Topic'].unique())

        summarised_topics = neg_wot_metadata['summarised_topics'][1:]

        topics_frequency = neg_wot_metadata['topics_over_time'][['Topic','Frequency']].groupby('Topic').sum().reset_index()
        wot_negative_feedback_frequency_df = \
            summarised_topics \
            .merge(
                topics_frequency,
                on = 'Topic'
                ) \
            .rename(
                columns={
                            'CustomName':'Topic Theme',
                            'Representation':'Top 10 Most Frequent Words',
                            'Count':'Occurences'
                        }
            )
        top_topics = st.slider('Select the number of most frequent topics...', 1, wot_max_topics, 10)
        st.write(visualize_topics_over_time(neg_wot_metadata,neg_wot_metadata['topics_over_time'], top_n_topics=top_topics,custom_labels=True))
        
        topic_option = st.selectbox(
            "Select a topic...",
            tuple(wot_negative_feedback_frequency_df.head(top_topics)['Topic Theme'].values),
            )
        # import pdb;pdb.set_trace()
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(wot_negative_feedback_frequency_df[['Topic Theme','Frequency']].head(top_topics),use_container_width=True)
        with col2:
            top_10_words = wot_negative_feedback_frequency_df[wot_negative_feedback_frequency_df['Topic Theme']==topic_option]['Top 10 Most Frequent Words'].values[0]
            st.write(top_10_words)
        
        
        st.write(visualize_documents(neg_wot_metadata,neg_wot_metadata['data']['review'].values, reduced_embeddings=neg_wot_metadata['reduced_embeddings'], custom_labels=True, hide_annotations=True,width=1250))
        st.write(visualize_hierarchy(neg_wot_metadata,orientation='left',custom_labels=True,top_n_topics=top_topics,width=1250))
    with tab_pos:
        st.write("Positive Feedback")
        # Get model and metadata from World of Tanks
        # pos_wot_model = pos_models['World of Tanks Blitz']
        pos_wot_metadata = pos_metadata['World of Tanks Blitz']
        wot_max_topics = max(pos_wot_metadata['topics_over_time']['Topic'].unique())

        # import pdb;pdb.set_trace()
        pos_summarised_topics = pos_wot_metadata['summarised_topics']

        pos_topics_frequency = pos_wot_metadata['topics_over_time'][['Topic','Frequency']].groupby('Topic').sum().reset_index()
        wot_positive_feedback_frequency_df = \
            pos_summarised_topics \
            .merge(
                pos_topics_frequency,
                on = 'Topic'
                ) \
            .rename(
                columns={
                            'CustomName':'Topic Theme',
                            'Representation':'Top 10 Most Frequent Words',
                            'Count':'Occurences'
                        }
            )

        top_topics_pos = st.slider('Select the number of most frequent topics... ', 1, wot_max_topics, 10)
        st.write(visualize_topics_over_time(pos_wot_metadata,pos_wot_metadata['topics_over_time'], top_n_topics=top_topics_pos,custom_labels=True))

        topic_option = st.selectbox(
            "Select a topic...",
            tuple(wot_positive_feedback_frequency_df.head(top_topics_pos)['Topic Theme'].values),
            )

        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(wot_positive_feedback_frequency_df[['Topic Theme','Frequency']].head(top_topics_pos),use_container_width=True)
        with col2:
            top_10_words = wot_positive_feedback_frequency_df[wot_positive_feedback_frequency_df['Topic Theme']==topic_option]['Top 10 Most Frequent Words'].values[0]
            st.write(top_10_words)
        st.write(visualize_documents(pos_wot_metadata,pos_wot_metadata['data']['review'].values, reduced_embeddings=pos_wot_metadata['reduced_embeddings'], custom_labels=True, hide_annotations=True,width=1250))
        st.write(visualize_hierarchy(pos_wot_metadata,orientation='left',custom_labels=True,top_n_topics=top_topics_pos,width=1250))
    with tab_compare:
        st.write("Hello")
        comp_top_topics = st.slider('Select the number of most frequent topics...  ', 1, 50, 10)
        comp_col1,comp_col2 = st.columns(2)
        with comp_col1:
            st.write(visualize_topics_over_time(neg_wot_metadata,neg_wot_metadata['topics_over_time'], top_n_topics=comp_top_topics,custom_labels=True, width = 600))
        with comp_col2:
            st.write(visualize_topics_over_time(neg_wot_metadata,neg_wot_metadata['topics_over_time'], top_n_topics=comp_top_topics,custom_labels=True, width = 600))
           



    

if __name__ == "__main__":
    main()