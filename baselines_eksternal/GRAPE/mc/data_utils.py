train_edge_attr = soft_one_hot(train_indices, len(class_values))
        train_edge_attr = torch.cat((train_edge_attr, train_edge_attr),0)
    else:
        train_edge_attr = torch.tensor(np.append(train_labels,train_labels)[:,None],
                            dtype=torch.float)

    val_v_indices = val_v_indices + n_row
    val_edge_index = torch.tensor([np.append(val_u_indices,val_v_indices),
                        np.append(val_v_indices,val_u_indices)],
                        dtype=int)
    if one_hot_edge:
        val_edge_attr = one_hot(val_indices, len(class_values))
        val_edge_attr = torch.cat((val_edge_attr, val_edge_attr),0)
    elif soft_one_hot_edge:
        val_edge_attr = soft_one_hot(val_indices, len(class_values))
        val_edge_attr = torch.cat((val_edge_attr, val_edge_attr),0)
    else:
        val_edge_attr = torch.tensor(np.append(val_labels,val_labels)[:,None],
                            dtype=torch.float)

    test_v_indices = test_v_indices + n_row
    test_edge_index = torch.tensor([np.append(test_u_indices,test_v_indices),
                        np.append(test_v_indices,test_u_indices)],
                        dtype=int)
    if one_hot_edge:
        test_edge_attr = one_hot(test_indices, len(class_values))
        test_edge_attr = torch.cat((test_edge_attr, test_edge_attr),0)
    elif soft_one_hot_edge:
        test_edge_attr = soft_one_hot(test_indices, len(class_values))
        test_edge_attr = torch.cat((test_edge_attr, test_edge_attr),0)
    else:
        test_edge_attr = torch.tensor(np.append(test_labels,test_labels)[:,None],
                            dtype=torch.float)
        
    if ce_loss:
        train_labels = torch.tensor(train_indices,dtype=int)
        val_labels = torch.tensor(val_indices,dtype=int)
        test_labels = torch.tensor(test_indices,dtype=int)

    data = Data(x=x,
            train_edge_index=train_edge_index,train_edge_attr=train_edge_attr,train_labels=train_labels,
            val_edge_index=val_edge_index,val_edge_attr=val_edge_attr,val_labels=val_labels,
            test_edge_index=test_edge_index,test_edge_attr=test_edge_attr,test_labels=test_labels,
            edge_attr_dim=train_edge_attr.shape[-1], class_values=torch.FloatTensor(class_values),
            user_num=adj_train.shape[0]
            )
    return data

# def split_edge(edge_attr,edge_index,batch_size):
#     edge_num = int(edge_attr.shape[0]/2)
#     perm = np.random.permutation(edge_num)
#     edge_attrs, edge_indexes = [],[]
#     index = 0
#     while index + batch_size < edge_num:
#         edge_attr_i = torch.cat((edge_attr[index:index+batch_size,:],
#                                 edge_attr[index:index+batch_size,:]),dim=0)
#         edge_start_i = torch.cat([edge_index[0,index:index+batch_size],
#                                     edge_index[1,index:index+batch_size]])
#         edge_end_i = torch.cat([edge_index[1,index:index+batch_size],
#                                     edge_index[0,index:index+batch_size]])
#         edge_index_i = torch.tensor([edge_start_i.detach().numpy(),
#                                     edge_end_i.detach().numpy()],
#                                     dtype=int)
#         index += batch_size
#         edge_attrs.append(edge_attr_i)
#         edge_indexes.append(edge_index_i)
#     return edge_attrs, edge_indexes

def load_data(args):
    mc_path = osp.dirname(osp.abspath(inspect.getfile(inspect.currentframe())))
    rating_map, post_rating_map = None, None
    if args.standard_rating:
        if args.data in ['flixster', 'ml_10m']: # original 0.5, 1, ..., 5
            rating_map = {x: int(math.ceil(x)) for x in np.arange(0.5, 5.01, 0.5).tolist()}
        elif args.data == 'yahoo_music':  # original 1, 2, ..., 100
            rating_map = {x: (x-1)//20+1 for x in range(1, 101)}
        else:
            rating_map = None

    if args.data == 'ml_1m' or args.data == 'ml_10m':
        if args.use_features:
            datasplit_path = mc_path+'/raw_data/' + args.data + '/withfeatures_split_seed' + str(args.data_seed) + '.pickle'
        else:
            datasplit_path = mc_path+'/raw_data/' + args.data + '/split_seed' + str(args.data_seed) + '.pickle'
    elif args.use_features:
        datasplit_path = mc_path+'/raw_data/' + args.data + '/withfeatures.pickle'
    else:
        datasplit_path = mc_path+'/raw_data/' + args.data + '/nofeatures.pickle'

    if args.data == 'flixster' or args.data == 'douban' or args.data == 'yahoo_music':
        u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices, \
            val_labels, val_u_indices, val_v_indices, test_labels, \
            test_u_indices, test_v_indices, class_values = load_data_monti(args.data, args.testing, rating_map, post_rating_map)
    elif args.data == 'ml_100k':
        print("Using official MovieLens dataset split u1.base/u1.test with 20% validation set size...")
        u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices, \
            val_labels, val_u_indices, val_v_indices, test_labels, \
            test_u_indices, test_v_indices, class_values = load_official_trainvaltest_split(args.data, args.testing, rating_map, post_rating_map, args.ratio)
    else:
        print("Using random dataset split ...")
        u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices, \
            val_labels, val_u_indices, val_v_indices, test_labels, \
            test_u_indices, test_v_indices, class_values = create_trainvaltest_split(args.data, args.data_seed, args.testing, datasplit_path, True, True, rating_map, post_rating_map, args.ratio)

    print('All ratings are:')
    print(class_values)
    '''
    Explanations of the above preprocessing:
        class_values are all the original continuous ratings, e.g. 0.5, 2...
        They are transformed to rating labels 0, 1, 2... acsendingly.
        Thus, to get the original rating from a rating label, apply: class_values[label]
        Note that train_labels etc. are all rating labels.
        But the numbers in adj_train are rating labels + 1, why? Because to accomodate neutral ratings 0! Thus, to get any edge label from adj_train, remember to substract 1.
        If testing=True, adj_train will include both train and val ratings, and all train data will be the combination of train and val.
    '''

    if args.use_features:
        u_features, v_features = u_features.toarray(), v_features.toarray()
        n_features = u_features.shape[1] + v_features.shape[1]
        print('Number of user features {}, item features {}, total features {}'.format(u_features.shape[1], v_features.shape[1], n_features))
    else:
        u_features, v_features = None, None
        n_features = 0

    if args.debug:  # use a small number of data to debug
        num_data = 1000
        train_u_indices, train_v_indices = train_u_indices[:num_data], train_v_indices[:num_data]
        train_labels = train_labels[:num_data]
        val_u_indices, val_v_indices = val_u_indices[:num_data], val_v_indices[:num_data]
        val_labels = val_labels[:num_data]
        test_u_indices, test_v_indices = test_u_indices[:num_data], test_v_indices[:num_data]
        test_labels = test_labels[:num_data]

    print('#train: %d, #val: %d, #test: %d' % (len(train_u_indices), len(val_u_indices), len(test_u_indices)))

    '''
        Transfer to torch geometric Data
    '''
    if not hasattr(args,'soft_one_hot_edge'):
        args.soft_one_hot_edge = False
    data = get_data(u_features, v_features, args.node_mode, adj_train,
        train_labels, train_u_indices, train_v_indices, \
        val_labels, val_u_indices, val_v_indices, test_labels, \
        test_u_indices, test_v_indices, class_values,
        args.one_hot_edge, args.soft_one_hot_edge, args.norm_label, args.ce_loss)
    return data

ypes, engine='python')

        # shuffle here like cf-nade paper with python's own random class
        # make sure to convert to list, otherwise random.shuffle acts weird on it without a warning
        data_array = data.values.tolist()
        random.seed(seed)
        random.shuffle(data_array)
        data_array = np.array(data_array)

        u_nodes_ratings = data_array[:, 0].astype(dtypes['u_nodes'])
        v_nodes_ratings = data_array[:, 1].astype(dtypes['v_nodes'])
        ratings = data_array[:, 2].astype(dtypes['ratings'])

        u_nodes_ratings, u_dict, num_users = map_data(u_nodes_ratings)
        v_nodes_ratings, v_dict, num_items = map_data(v_nodes_ratings)

        u_nodes_ratings, v_nodes_ratings = u_nodes_ratings.astype(np.int64), v_nodes_ratings.astype(np.int64)
        ratings = ratings.astype(np.float32)

        # Load movie features
        movies_file = data_dir + files[1]

        movies_headers = ['movie_id', 'title', 'genre']
        movies_df = pd.read_csv(movies_file, sep=sep, header=None,
                                names=movies_headers, engine='python')

        # Extracting all genres
        genres = []
        for s in movies_df['genre'].values:
            genres.extend(s.split('|'))

        genres = list(set(genres))
        num_genres = len(genres)

        genres_dict = {g: idx for idx, g in enumerate(genres)}

        # Creating 0 or 1 valued features for all genres
        v_features = np.zeros((num_items, num_genres), dtype=np.float32)
        for movie_id, s in zip(movies_df['movie_id'].values.tolist(), movies_df['genre'].values.tolist()):
            # Check if movie_id was listed in ratings file and therefore in mapping dictionary
            if movie_id in v_dict.keys():
                gen = s.split('|')
                for g in gen:
                    v_features[v_dict[movie_id], genres_dict[g]] = 1.

        # Load user features
        users_file = data_dir + files[2]
        users_headers = ['user_id', 'gender', 'age', 'occupation', 'zip-code']
        users_df = pd.read_csv(users_file, sep=sep, header=None,
                               names=users_headers, engine='python')

        # Extracting all features
        cols = users_df.columns.values[1:]

        cntr = 0
        feat_dicts = []
        for header in cols:
            d = dict()
            feats = np.unique(users_df[header].values).tolist()
            d.update({f: i for i, f in enumerate(feats, start=cntr)})
            feat_dicts.append(d)
            cntr += len(d)

        num_feats = sum(len(d) for d in feat_dicts)

        u_features = np.zeros((num_users, num_feats), dtype=np.float32)
        for _, row in users_df.iterrows():
            u_id = row['user_id']
            if u_id in u_dict.keys():
                for k, header in enumerate(cols):
                    u_features[u_dict[u_id], feat_dicts[k][row[header]]] = 1.

        u_features = sp.csr_matrix(u_features)
        v_features = sp.csr_matrix(v_features)

    elif fname == 'ml_10m':

        # Check if files exist and download otherwise
        files = ['/ratings.dat']
        download_dataset(fname, files, data_dir)

        sep = r'\:\:'

        filename = data_dir + files[0]

        dtypes = {
            'u_nodes': np.int64, 'v_nodes': np.int64,
            'ratings': np.float32, 'timestamp': np.float64}

        # use engine='python' to ignore warning about switching to python backend when using regexp for sep
        data = pd.read_csv(filename, sep=sep, header=None,
                           names=['u_nodes', 'v_nodes', 'ratings', 'timestamp'], converters=dtypes, engine='python')

        # shuffle here like cf-nade paper with python's own random class
        # make sure to convert to list, otherwise random.shuffle acts weird on it without a warning
        data_array = data.values.tolist()
        random.seed(seed)
        random.shuffle(data_array)
        data_array = np.array(data_array)

        u_nodes_ratings = data_array[:, 0].astype(dtypes['u_nodes'])
        v_nodes_ratings = data_array[:, 1].astype(dtypes['v_nodes'])
        ratings = data_array[:, 2].astype(dtypes['ratings'])

        u_nodes_ratings, u_dict, num_users = map_data(u_nodes_ratings)
        v_nodes_ratings, v_dict, num_items = map_data(v_nodes_ratings)

        u_nodes_ratings, v_nodes_ratings = u_nodes_ratings.astype(np.int64), v_nodes_ratings.astype(np.int64)
        ratings = ratings.astype(np.float32)

    else:
        raise ValueError('Dataset name not recognized: ' + fname)

    if verbose:
        print('Number of users = %d' % num_users)
        print('Number of items = %d' % num_items)
        print('Number of links = %d' % ratings.shape[0])
        print('Fraction of positive links = %.4f' % (float(ratings.shape[0]) / (num_users * num_items),))

    return num_users, num_items, u_nodes_ratings, v_nodes_ratings, ratings, u_features, v_features
