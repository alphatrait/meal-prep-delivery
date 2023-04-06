import pandas as pd
import requests
import json
import numpy as np
from k_means_constrained import KMeansConstrained
from math import sqrt
from itertools import product
from urllib.request import urlopen
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


API_KEY_GC = 'Google API Key for Geocoding'
API_KEY_DM = 'Google API Key for Distance Matrix'


def optimizer(df):
    response_out = []
    pd.set_option('display.max_rows', None)
    df.index.name = 'no'

    geocoded = []

    for i, row in df.iterrows():
        apiAddress = str(df.at[i,'street'])+','+str(df.at[i,'zip'])+','+str(df.at[i,'city'])+','+str(df.at[i,'country'])    
        parameters = {
            "address" : apiAddress,
            "key" : API_KEY_GC
            
        }

        response = requests.get("https://maps.googleapis.com/maps/api/geocode/json", params=parameters)
        data = response.text
        dataJ = json.loads(data)['results']

        lat = (dataJ[0]['geometry']['location']['lat'])
        lng = (dataJ[0]['geometry']['location']['lng'])
        
        df.at[i,'lat'] = lat
        df.at[i,'lng'] = lng

        geocoded.append([lat, lng])



    # ---------------------- #
    # -------Cluster-------- #
    # ---------------------- #


    X = np.array(geocoded)

    df['lat'] = X[:,0]
    df['lng'] = X[:,1]

    total_packages = len(df)
    min_capacity = 7
    drivers_needed = 1

    if total_packages < 15:
        max_capacity = total_packages
    else:
        max_capacity = 15

    if  total_packages < 9:
        drivers_needed = 1
    else:
        for i in range(9, total_packages, 10):
            drivers_needed += 1

    if drivers_needed == 1:
        min_capacity = 1
        df['cluster'] = 0
        print ("Only one driver needed")
        
    else:
        min_capacity = 5

        clf = KMeansConstrained(
            n_clusters = drivers_needed,
            size_min = min_capacity,
            size_max= max_capacity,
            random_state = 0
        )

        clf.fit_predict(X)

        # save results
        labels = clf.labels_

        # send back into dataframe and display it
        df['cluster'] = labels

        # display the number of mamber each clustering
        _clusters = df.groupby('cluster').count()

        print ("You need ", drivers_needed, " drivers")



    # --------------------- #
    # -------Worker-------- #
    # --------------------- #

    for i in df['cluster'].unique():
        origin = {
            'street': df['street'][0],
            'zip': df['zip'][0],
            'city': df['city'][0],
            'country': df['country'][0],
            'order_id': 0,
            'lat': df['lat'][0],
            'lng': df['lng'][0],
            'cluster': i
        }

        print(origin)

        df.loc[-0.5] = origin
        df = df.sort_index().reset_index(drop=True)

        coordinates = []

        for row in df.iterrows():
            if row[-1]['cluster'] == i:
                coordinates.append([float(row[1]['lat']), float(row[1]['lng'])])

        def generate_data():
            data = {
                'API_key': API_KEY_DM,
                'coordinates': coordinates
            }
            return data

        def get_coordinates_matrix(coordinates_list):
            '''Generate the matrix of coordinates'''
            coordinates_product = list(product(coordinates_list, repeat=2))
            matrix = []
            for prod in coordinates_product:
                row = []
                for item in prod:
                    row.extend(item)
                matrix.append(row)
            return matrix

        def generate_requests_list(API_key, coordinates_matrix):
            '''Generate the list of request strings'''
            requests_list = []
            for coords in coordinates_matrix:
                base = 'https://maps.googleapis.com/maps/api/distancematrix/json?origins='
                requests_list.append(f'{base}{coords[0]}%2C{coords[1]}&destinations={coords[2]}%2C{coords[3]}&key={API_key}')
            return requests_list

        def send_request(request):
            '''Get response by sending request'''
            json_result = urlopen(request).read()
            response = json.loads(json_result)
            return response

        def get_distance_matrix(requests):
            'Print the responses of all requests'
            rows_count = int(sqrt(len(requests)))
            matrix = []
            row = []
            for index, request in enumerate(requests):
                response = send_request(request)
                row.append(response['rows'][0]['elements'][0]['distance']['value'])
                if (index + 1) % rows_count == 0:
                    matrix.append(row)
                    row = []
            return matrix

        def worker():
            '''Main function to run the entire code'''
            data = generate_data()
            coordinates_matrix = get_coordinates_matrix(data['coordinates'])
            requests_list = generate_requests_list(API_key=data['API_key'], coordinates_matrix=coordinates_matrix)
            # for row in get_distance_matrix(requests_list):
            #     print(row)
            work = get_distance_matrix(requests_list)
            df = pd.DataFrame(get_distance_matrix(requests_list))
            return work

        worker()
        #if __name__ == '__main__':
        #    worker()



    # --------------------- #
    # -------Solver-------- #
    # --------------------- #


        def create_data_model():
            '''Generate the data'''
            data = {}
            data['distance_matrix'] = worker()
            data['num_vehicles'] = 1
            data['depot'] = 0
            return data


        def print_solution(data, manager, routing, solution):
            """Prints solution on console."""
            print(f'Objective: {solution.ObjectiveValue()}')
            for vehicle_id in range(data['num_vehicles']):
                index = routing.Start(vehicle_id)
                plan_output = []
                route_distance = 0
                new_df = df.loc[df['cluster'] == i]
                while not routing.IsEnd(index):
                    address = new_df.iloc[manager.IndexToNode(index)]
                    plan_output.append({
                        "lat": address['lat'],
                        "lng": address['lng'],
                        "order_id": address['order_id']
                    })
                    previous_index = index
                    index = solution.Value(routing.NextVar(index))
                    route_distance += routing.GetArcCostForVehicle(
                        previous_index, index, vehicle_id)

                response_out.append({
                    "driver": int(i), 
                    "route": plan_output
                })     
            

        def main():
            """Entry point of the program."""
            # Instantiate the data problem.
            data = create_data_model()

            # Create the routing index manager.
            manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                                data['num_vehicles'], data['depot'])

            # Create Routing Model.
            routing = pywrapcp.RoutingModel(manager)


            # Create and register a transit callback.
            def distance_callback(from_index, to_index):
                """Returns the distance between the two nodes."""
                # Convert from routing variable Index to distance matrix NodeIndex.
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                return data['distance_matrix'][from_node][to_node]

            transit_callback_index = routing.RegisterTransitCallback(distance_callback)

            # Define cost of each arc.
            routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

            # Add Distance constraint.
            dimension_name = 'Distance'
            routing.AddDimension(
                transit_callback_index,
                0,  # no slack
                500000,  # vehicle maximum travel distance
                True,  # start cumul to zero
                dimension_name)
            distance_dimension = routing.GetDimensionOrDie(dimension_name)
            distance_dimension.SetGlobalSpanCostCoefficient(100)

            # Setting first solution heuristic.
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            search_parameters.first_solution_strategy = (
                routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

            # Solve the problem.
            solution = routing.SolveWithParameters(search_parameters)

            # Print solution on console.
            if solution:
                print_solution(data, manager, routing, solution)
            else:
                print('No solution found !')

        main()
        #if __name__ == '__main__':
        #    main()
    return response_out

if __name__=="__main__":
    df = pd.read_csv("sample-locations.csv", on_bad_lines='skip')
    print("Output from the API")
    print(optimizer(df))