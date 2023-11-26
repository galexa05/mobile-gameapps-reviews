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

@st.cache_data
def get_top_10_themes(topic_data):
    summarised_topics = topic_data['summarised_topics'][1:]

    topics_frequency = topic_data['topics_over_time'][['Topic','Frequency']].groupby('Topic').sum().reset_index()
    topic_data_frequency = \
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
    return topic_data_frequency[['Topic Theme','Frequency']].head(10)

def main():
    # Set the title and header of the app
    st.title("Gaming Insights Dashboard: Unveiling Wargaming and Competitor Reviews")
    st.header("Main Company Focus: Wargaming Overview & Comparative Analysis of Competitor Reviews")
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
        
        st.markdown(
            """
               #### Monthly Evolution of Top Negative Themes for the World of Tanks Blitz
               **Description**: \n
               Explore the changing landscape of negative themes over time through this interactive time-series graph. 
               The x-axis represents the monthly timeline, while the y-axis shows the frequency of selected negative themes. 
               Customize your view by selecting the top n most negative themes using the filter parameter. 
               Higher peaks indicate increased prevalence of these themes, offering a straightforward way to observe patterns 
               and trends.
            """
        )
        st.write(visualize_topics_over_time(neg_wot_metadata,neg_wot_metadata['topics_over_time'], top_n_topics=top_topics,custom_labels=True))
        
        st.markdown(
            """
            #### Overview of the top N Negative Themes for the World of Tanks Blitz
            Delve into negative themes with the following graphs. On the left side, you'll find the top N negative theme titles and see 
            how often they appear in overall. 
            Use the filter to choose a specific title, and on the right, discover the representative word for that theme.
            It's an easy way to grasp the big picture of negative themes and zoom in on what matters to you.
            """
        )
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
        
        st.markdown(
            """
            #### Explore Negative Experiences in Detail: Interactive 2D Map for the World of Tanks Blitz
            Dive into the realm of less-than-ideal reviews with this 2D map. Each color represents a different negative topic, 
            and every point on the map is an individual negative review. Hover over these points to reveal specific details and gain 
            a deeper understanding of each negative experience. The map not only shows which negative topics are close and relevant 
            but also how they differ in their contexts. Uncover the intricacies of negative feedback, understanding not only what's important 
            but also the specific details that shape users' less favorable experiences. It's an enlightening exploration into the 
            landscape of challenges, providing clear insights and practical ideas for addressing concerns and enhancing overall 
            user satisfaction.
            """        
            )
        st.write(visualize_documents(neg_wot_metadata,neg_wot_metadata['data']['review'].values, reduced_embeddings=neg_wot_metadata['reduced_embeddings'], custom_labels=True, hide_annotations=True,width=1250))
        st.markdown(
            """
            #### Explore the Connections Among Negative Topics: Hierarchical Path Chart for the World of Tanks Blitz
            Navigate through the web of negative topics with our hierarchical path chart. Each node in the chart represents 
            a negative topic, and the lines depict the relevance and connections among them. Follow the branches to understand 
            how these negative topics are interlinked, revealing a tree-like structure. This straightforward visualization enables 
            you to grasp the relationships and dependencies between negative themes, offering a clear overview of their connections.
            """        
            )        
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

        st.markdown(
            """
               #### Monthly Evolution of Top Positive Themes for the World of Tanks Blitz
               **Description**: \n
               Explore the changing landscape of positive themes over time through this interactive plot. 
               The x-axis reveals the monthly timeline, while the y-axis illustrates the frequency of selected positive themes.
               Customize your view by using the filter parameter to highlight the top n most positive themes. 
               Notice peaks indicating periods of increased prevalence, offering insights into evolving patterns and trends.
               Gain a deeper understanding of the positive thematic landscape without the need for specialized knowledge.
            """
        )

        st.write(visualize_topics_over_time(pos_wot_metadata,pos_wot_metadata['topics_over_time'], top_n_topics=top_topics_pos,custom_labels=True))
        
        st.markdown(
            """
            #### Overview of the top N Positive Themes for the World of Tanks Blitz
            Delve into positive themes with the following graphs. On the left side, explore the 
            top N positive theme titles and see how often they appear overall. Use the filter to pick a specific title, 
            and on the right, uncover the representative word for that theme. It's a straightforward way to understand the 
            broader picture of positive themes and zoom in on what interests you.
            """
        )
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

        st.markdown(
            """
            #### Explore Positive Experiences in Detail: Interactive 2D Map for the World of Tanks Blitz
            Dive into the world of happy reviews with this 2D map. Each color shows a different positive topic, 
            and every point on the map is a single positive review. Hover over these points to discover specific details 
            and get a deeper understanding of each positive experience. The map not only shows which positive topics are close 
            and relevant but also how they differ in their situations. Uncover the details of positive feedback, 
            figuring out not only what's important but also the little things that make users happy.             """
        )
        st.write(visualize_documents(pos_wot_metadata,pos_wot_metadata['data']['review'].values, reduced_embeddings=pos_wot_metadata['reduced_embeddings'], custom_labels=True, hide_annotations=True,width=1250))
        
        st.markdown(
            """
            #### Explore the Connections Among Negative Topics: Hierarchical Path Chart for the World of Tanks Blitz
            Navigate through the web of positive topics with our hierarchical path chart. Each node in the chart represents 
            a positive topic, and the lines depict the relevance and connections among them. Follow the branches to understand 
            how these positive topics are interlinked, revealing a tree-like structure. This straightforward visualization enables 
            you to grasp the relationships and dependencies between positive themes, offering a clear overview of their connections.            """        
            )  
        st.write(visualize_hierarchy(pos_wot_metadata,orientation='left',custom_labels=True,top_n_topics=top_topics_pos,width=1250))
    
    
    with tab_compare:
        feedback_option = st.selectbox(
            'Select Negative or Positive feedback comparison...',
            ('Positive','Negative'),
            index=0
        )
        if feedback_option=='Positive':
            selected_data = pos_metadata
        else:
            selected_data = neg_metadata

        game_apps = tuple(neg_metadata.keys())
        comp_app1,comp_app2 = st.columns(2)
        with comp_app1:
            game_option1 = st.selectbox(
            'Select the 1st game app',
            game_apps,
            index=7
            )
        with comp_app2:
            game_option2 = st.selectbox(
            'Select the 2nd game app',
            game_apps,
            index=0
            )
        app_data1 = selected_data[game_option1]
        app_data2 = selected_data[game_option2]  
        app_data1_maxtopics = max(app_data1['topics_over_time']['Topic'].unique())
        app_data2_maxtopics = max(app_data2['topics_over_time']['Topic'].unique())

        st.markdown(f"<center><h4>Top 10 {feedback_option} Topic themes between <b>{game_option1}</b> and <b>{game_option2}</b></h4></center>",unsafe_allow_html=True)
        comp_col1,comp_col2 = st.columns(2)
        with comp_col1:
            st.markdown(f"#### {game_option1}")
            st.dataframe(get_top_10_themes(app_data1),use_container_width=True)
        with comp_col2:
            st.markdown(f"#### {game_option2}")
            st.dataframe(get_top_10_themes(app_data2),use_container_width=True)
        
        st.markdown(f"<center><h4>Top 10 {feedback_option} Topic themes over time between <b>{game_option1}</b> and <b>{game_option2}</b></h4></center>",unsafe_allow_html=True)
        time_col1,time_col2 = st.columns(2)
        with time_col1:
            st.markdown(f"#### {game_option1}")
            st.write(visualize_topics_over_time(app_data1,app_data1['topics_over_time'], top_n_topics=10,custom_labels=True, width = 800))
        with time_col2:
            st.markdown(f"#### {game_option2}")
            st.write(visualize_topics_over_time(app_data2,app_data2['topics_over_time'], top_n_topics=10,custom_labels=True, width = 800))

        st.markdown(f"<center><h4>Visualize {feedback_option} topics</h4></center>",unsafe_allow_html=True)
        vis_topics_col1,vis_topics_col2 = st.columns(2)
        with vis_topics_col1:
            st.write(visualize_topics(app_data1,custom_labels=True,title=f"Interactive 2D Map of {feedback_option} topics ({game_option1})"))
        with vis_topics_col2:
            st.write(visualize_topics(app_data2,custom_labels=True,title=f"Interactive 2D Map of {feedback_option} topics ({game_option2}"))

        st.markdown(f"<center><h4>Visualize the hierarchy between {feedback_option} topics</h4></center>",unsafe_allow_html=True)
        hierarchy_topics_col1,hierarchy_topics_col2 = st.columns(2)
        with hierarchy_topics_col1:
            hier_most_topics1 = st.slider(f'Select the N most {feedback_option} frequent topics...    ', 1, app_data1_maxtopics, 30)
            st.write(visualize_hierarchy(app_data1,orientation='left',custom_labels=True,top_n_topics=hier_most_topics1,width=700,title=f"Hierarchical Clustering of Topics ({game_option1})"))
        with hierarchy_topics_col2:
            hier_most_topics2 = st.slider(f'Select the N most {feedback_option} frequent topics...     ', 1, app_data2_maxtopics, 30)
            st.write(visualize_hierarchy(app_data2,orientation='left',custom_labels=True,top_n_topics=hier_most_topics2,width=700,title=f"Hierarchical Clustering of Topics ({game_option2})"))

        st.markdown(f"<center><h4>Visualize grouped {feedback_option} reviews per topic theme in 2D Map</h4></center>",unsafe_allow_html=True)
        review_topics_col1,review_topics_col2 = st.columns(2)
        with review_topics_col1:
            st.write(visualize_documents(app_data1,app_data1['data']['review'].values, reduced_embeddings=app_data1['reduced_embeddings'], custom_labels=True, hide_annotations=True,width=700, title=f"Interactive 2D Map of {feedback_option} reviews ({game_option1})"))        
        with review_topics_col2:
            st.write(visualize_documents(app_data2,app_data2['data']['review'].values, reduced_embeddings=app_data2['reduced_embeddings'], custom_labels=True, hide_annotations=True,width=700, title=f"Interactive 2D Map of {feedback_option} reviews ({game_option2})"))


    

if __name__ == "__main__":
    main()