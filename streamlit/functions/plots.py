import pandas as pd
from typing import List,Union
from sklearn.preprocessing import normalize
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np
from typing import Callable, List, Union
from scipy.sparse import csr_matrix
from scipy.cluster import hierarchy as sch
from sklearn.metrics.pairwise import cosine_similarity
from bertopic._utils import validate_distance_matrix


def visualize_topics_over_time(
                               topic_metadata,
                               topics_over_time: pd.DataFrame,
                               top_n_topics: int = None,
                               topics: List[int] = None,
                               normalize_frequency: bool = False,
                               custom_labels: Union[bool, str] = False,
                               title: str = "<b>Topics over Time</b>",
                               width: int = 1250,
                               height: int = 450) -> go.Figure:

    colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#D55E00", "#0072B2", "#CC79A7"]

    # Select topics based on top_n and topics args
    freq_df = topic_metadata['topics_over_time'].groupby('Topic').sum('Frequency').rename(columns={'Frequency':'Count'}).reset_index()
    freq_df = freq_df.loc[freq_df.Topic != -1, :]
    if topics is not None:
        selected_topics = list(topics)
    elif top_n_topics is not None:
        selected_topics = sorted(freq_df.Topic.to_list()[:top_n_topics])
    else:
        selected_topics = sorted(freq_df.Topic.to_list())

    # Prepare data
    topic_names = {row['Topic']:row['CustomName'] for _,row in topic_metadata['summarised_topics'].iterrows()}
    topics_over_time["Name"] = topics_over_time.Topic.map(topic_names)
    data = topics_over_time.loc[topics_over_time.Topic.isin(selected_topics), :].sort_values(["Topic", "Timestamp"])

    # Add traces
    fig = go.Figure()
    for index, topic in enumerate(data.Topic.unique()):
        trace_data = data.loc[data.Topic == topic, :]
        topic_name = trace_data.Name.values[0]
        words = trace_data.Words.values
        if normalize_frequency:
            y = normalize(trace_data.Frequency.values.reshape(1, -1))[0]
        else:
            y = trace_data.Frequency
        fig.add_trace(go.Scatter(x=trace_data.Timestamp, y=y,
                                 mode='lines',
                                 marker_color=colors[index % 7],
                                 hoverinfo="text",
                                 name=topic_name,
                                 hovertext=[f'<b>Topic {topic}</b><br>Words: {word}' for word in words]))

    # Styling of the visualization
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    fig.update_layout(
        yaxis_title="Normalized Frequency" if normalize_frequency else "Frequency",
        title={
            'text': f"{title}",
            'y': .95,
            'x': 0.40,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
        template="simple_white",
        width=width,
        height=height,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
        legend=dict(
            title="<b>Global Topic Representation",
        )
    )
    return fig


def visualize_documents(
                        topic_metadata,
                        docs: List[str],
                        topics: List[int] = None,
                        embeddings: np.ndarray = None,
                        reduced_embeddings: np.ndarray = None,
                        sample: float = None,
                        hide_annotations: bool = False,
                        hide_document_hover: bool = False,
                        custom_labels: Union[bool, str] = False,
                        title: str = "<b>Documents and Topics</b>",
                        width: int = 1200,
                        height: int = 750):
    topic_per_doc = list(topic_metadata['data']['Topic'])

    # Sample the data to optimize for visualization and dimensionality reduction
    if sample is None or sample > 1:
        sample = 1

    indices = []
    for topic in set(topic_per_doc):
        s = np.where(np.array(topic_per_doc) == topic)[0]
        size = len(s) if len(s) < 100 else int(len(s) * sample)
        indices.extend(np.random.choice(s, size=size, replace=False))
    indices = np.array(indices)

    df = pd.DataFrame({"topic": np.array(topic_per_doc)[indices]})
    df["doc"] = [docs[index] for index in indices]
    df["topic"] = [topic_per_doc[index] for index in indices]

    # Extract embeddings if not already done
    
    embeddings_2d = reduced_embeddings[indices]
    
    unique_topics = set(topic_per_doc)
    if topics is None:
        topics = unique_topics

    # Combine data
    df["x"] = embeddings_2d[:, 0]
    df["y"] = embeddings_2d[:, 1]

    # Prepare text and names

    summarised_topics_df = topic_metadata['summarised_topics']
    names = list(summarised_topics_df[summarised_topics_df['Topic']!=-1]['CustomName'])

    # Visualize
    fig = go.Figure()

    # Outliers and non-selected topics
    non_selected_topics = set(unique_topics).difference(topics)
    if len(non_selected_topics) == 0:
        non_selected_topics = [-1]

    selection = df.loc[df.topic.isin(non_selected_topics), :]
    selection["text"] = ""
    selection.loc[len(selection), :] = [None, None, selection.x.mean(), selection.y.mean(), "Other documents"]

    fig.add_trace(
        go.Scattergl(
            x=selection.x,
            y=selection.y,
            hovertext=selection.doc if not hide_document_hover else None,
            hoverinfo="text",
            mode='markers+text',
            name="other",
            showlegend=False,
            marker=dict(color='#CFD8DC', size=5, opacity=0.5)
        )
    )

    # Selected topics
    for name, topic in zip(names, unique_topics):
        if topic in topics and topic != -1:
            selection = df.loc[df.topic == topic, :]
            selection["text"] = ""

            if not hide_annotations:
                selection.loc[len(selection), :] = [None, None, selection.x.mean(), selection.y.mean(), name]

            fig.add_trace(
                go.Scattergl(
                    x=selection.x,
                    y=selection.y,
                    hovertext=selection.doc if not hide_document_hover else None,
                    hoverinfo="text",
                    text=selection.text,
                    mode='markers+text',
                    name=name,
                    textfont=dict(
                        size=12,
                    ),
                    marker=dict(size=5, opacity=0.5)
                )
            )

    # Add grid in a 'plus' shape
    x_range = (df.x.min() - abs((df.x.min()) * .15), df.x.max() + abs((df.x.max()) * .15))
    y_range = (df.y.min() - abs((df.y.min()) * .15), df.y.max() + abs((df.y.max()) * .15))
    fig.add_shape(type="line",
                  x0=sum(x_range) / 2, y0=y_range[0], x1=sum(x_range) / 2, y1=y_range[1],
                  line=dict(color="#CFD8DC", width=2))
    fig.add_shape(type="line",
                  x0=x_range[0], y0=sum(y_range) / 2, x1=x_range[1], y1=sum(y_range) / 2,
                  line=dict(color="#9E9E9E", width=2))
    fig.add_annotation(x=x_range[0], y=sum(y_range) / 2, text="D1", showarrow=False, yshift=10)
    fig.add_annotation(y=y_range[1], x=sum(x_range) / 2, text="D2", showarrow=False, xshift=10)

    # Stylize layout
    fig.update_layout(
        template="simple_white",
        title={
            'text': f"{title}",
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
        width=width,
        height=height
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def visualize_topics(
                     topic_metadata,
                     topics: List[int] = None,
                     top_n_topics: int = None,
                     custom_labels: Union[bool, str] = False,
                     title: str = "<b>Intertopic Distance Map</b>",
                     width: int = 650,
                     height: int = 650) -> go.Figure:
   
    # Select topics based on top_n and topics args
    # freq_df = topic_model.get_topic_freq()
    freq_df = topic_metadata['topics_over_time'].groupby('Topic').sum('Frequency').rename(columns={'Frequency':'Count'}).reset_index()
    freq_df = freq_df.loc[freq_df.Topic != -1, :]
    
    if topics is not None:
        topics = list(topics)
    elif top_n_topics is not None:
        topics = sorted(freq_df.Topic.to_list()[:top_n_topics])
    else:
        topics = sorted(freq_df.Topic.to_list())

    # Extract topic words and their frequencies
    topic_list = sorted(topics)
    # frequencies = [topic_model.topic_sizes_[topic] for topic in topic_list]
    frequencies = list(freq_df['Count'])
    summarised_topics_df = topic_metadata['summarised_topics']
    summarised_topics_df = summarised_topics_df[summarised_topics_df['Topic']!=-1]
    if custom_labels and topics is not None:
        # words = [topic_model.custom_labels_[topic + topic_model._outliers] for topic in topic_list]
        words = list(summarised_topics_df['CustomName'])
    else:
        # words = [" | ".join([word[0] for word in topic_model.get_topic(topic)[:5]]) for topic in topic_list]
        words = [" | ".join(word for word in topic[:5]) for topic in temp_data['summarised_topics']['Representation']]
        

    # Embed c-TF-IDF into 2D
    all_topics = topic_metadata['summarised_topics']['Topic'].values.tolist()
    indices = np.array([all_topics.index(topic) for topic in topics])
    embeddings = topic_metadata['topic_embeddings_2d']
    df = pd.DataFrame({"x": embeddings[:, 0], "y": embeddings[:, 1],
                       "Topic": topic_list, "Words": words, "Size": frequencies})
    return _plotly_topic_visualization(df, topic_list, title, width, height)


def _plotly_topic_visualization(df: pd.DataFrame,
                                topic_list: List[str],
                                title: str,
                                width: int,
                                height: int):
    """ Create plotly-based visualization of topics with a slider for topic selection """

    def get_color(topic_selected):
        if topic_selected == -1:
            marker_color = ["#B0BEC5" for _ in topic_list]
        else:
            marker_color = ["red" if topic == topic_selected else "#B0BEC5" for topic in topic_list]
        return [{'marker.color': [marker_color]}]

    # Prepare figure range
    x_range = (df.x.min() - abs((df.x.min()) * .15), df.x.max() + abs((df.x.max()) * .15))
    y_range = (df.y.min() - abs((df.y.min()) * .15), df.y.max() + abs((df.y.max()) * .15))

    # Plot topics
    fig = px.scatter(df, x="x", y="y", size="Size", size_max=40, template="simple_white", labels={"x": "", "y": ""},
                     hover_data={"Topic": True, "Words": True, "Size": True, "x": False, "y": False})
    fig.update_traces(marker=dict(color="#B0BEC5", line=dict(width=2, color='DarkSlateGrey')))

    # Update hover order
    fig.update_traces(hovertemplate="<br>".join(["<b>Topic %{customdata[0]}</b>",
                                                 "%{customdata[1]}",
                                                 "Size: %{customdata[2]}"]))

    # Create a slider for topic selection
    steps = [dict(label=f"Topic {topic}", method="update", args=get_color(topic)) for topic in topic_list]
    sliders = [dict(active=0, pad={"t": 50}, steps=steps)]

    # Stylize layout
    fig.update_layout(
        title={
            'text': f"{title}",
            'y': .95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
        width=width,
        height=height,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
        xaxis={"visible": False},
        yaxis={"visible": False},
        sliders=sliders
    )

    # Update axes ranges
    fig.update_xaxes(range=x_range)
    fig.update_yaxes(range=y_range)

    # Add grid in a 'plus' shape
    fig.add_shape(type="line",
                  x0=sum(x_range) / 2, y0=y_range[0], x1=sum(x_range) / 2, y1=y_range[1],
                  line=dict(color="#CFD8DC", width=2))
    fig.add_shape(type="line",
                  x0=x_range[0], y0=sum(y_range) / 2, x1=x_range[1], y1=sum(y_range) / 2,
                  line=dict(color="#9E9E9E", width=2))
    fig.add_annotation(x=x_range[0], y=sum(y_range) / 2, text="D1", showarrow=False, yshift=10)
    fig.add_annotation(y=y_range[1], x=sum(x_range) / 2, text="D2", showarrow=False, xshift=10)
    fig.data = fig.data[::-1]

    return fig



def visualize_hierarchy(
                        # topic_model,
                        topic_metadata,
                        orientation: str = "left",
                        topics: List[int] = None,
                        top_n_topics: int = None,
                        custom_labels: Union[bool, str] = False,
                        title: str = "<b>Hierarchical Clustering</b>",
                        width: int = 1000,
                        height: int = 600,
                        hierarchical_topics: pd.DataFrame = None,
                        linkage_function: Callable[[csr_matrix], np.ndarray] = None,
                        distance_function: Callable[[csr_matrix], csr_matrix] = None,
                        color_threshold: int = 1) -> go.Figure:

    if distance_function is None:
        distance_function = lambda x: 1 - cosine_similarity(x)

    if linkage_function is None:
        linkage_function = lambda x: sch.linkage(x, 'ward', optimal_ordering=True)

    # Select topics based on top_n and topics args
    freq_df = topic_metadata['topics_over_time'].groupby('Topic').sum('Frequency').rename(columns={'Frequency':'Count'}).reset_index()
    freq_df = freq_df.loc[freq_df.Topic != -1, :]
    if topics is not None:
        topics = list(topics)
    elif top_n_topics is not None:
        topics = sorted(freq_df.Topic.to_list()[:top_n_topics])
    else:
        topics = sorted(freq_df.Topic.to_list())

    # Select embeddings
    all_topics = topic_metadata['summarised_topics']['Topic'].values.tolist()
    indices = np.array([all_topics.index(topic) for topic in topics])
    embeddings = np.array(topic_metadata['topic_embeddings'])[indices] 
    print(embeddings.shape)   
    # Annotations
    annotations = None

    # wrap distance function to validate input and return a condensed distance matrix
    distance_function_viz = lambda x: validate_distance_matrix(
        distance_function(x), embeddings.shape[0])
    # Create dendogram
    fig = ff.create_dendrogram(embeddings,
                               orientation=orientation,
                               distfun=distance_function_viz,
                               linkagefun=linkage_function,
                               hovertext=annotations,
                               color_threshold=color_threshold)

    # Create nicer labels
    axis = "yaxis" if orientation == "left" else "xaxis"
    custom_labels_themes = topic_metadata['summarised_topics']['CustomName'].tolist()
    new_labels = [custom_labels_themes[topics[int(x)] + 1] for x in fig.layout[axis]["ticktext"]]

    # Stylize layout
    fig.update_layout(
        plot_bgcolor='#ECEFF1',
        template="plotly_white",
        title={
            'text': f"{title}",
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
    )

    # Stylize orientation
    if orientation == "left":
        fig.update_layout(height=200 + (15 * len(topics)),
                          width=width,
                          yaxis=dict(tickmode="array",
                                     ticktext=new_labels))

        # Fix empty space on the bottom of the graph
        y_max = max([trace['y'].max() + 5 for trace in fig['data']])
        y_min = min([trace['y'].min() - 5 for trace in fig['data']])
        fig.update_layout(yaxis=dict(range=[y_min, y_max]))

    else:
        fig.update_layout(width=200 + (15 * len(topics)),
                          height=height,
                          xaxis=dict(tickmode="array",
                                     ticktext=new_labels))

    if hierarchical_topics is not None:
        for index in [0, 3]:
            axis = "x" if orientation == "left" else "y"
            xs = [data["x"][index] for data in fig.data if (data["text"] and data[axis][index] > 0)]
            ys = [data["y"][index] for data in fig.data if (data["text"] and data[axis][index] > 0)]
            hovertext = [data["text"][index] for data in fig.data if (data["text"] and data[axis][index] > 0)]

            fig.add_trace(go.Scatter(x=xs, y=ys, marker_color='black',
                                     hovertext=hovertext, hoverinfo="text",
                                     mode='markers', showlegend=False))
    return fig


