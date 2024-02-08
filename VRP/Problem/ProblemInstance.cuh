//
// Created by Strut on 2/8/2024.
//

#ifndef VRP_PROBLEMINSTANCE_CUH
#define VRP_PROBLEMINSTANCE_CUH

#include <iostream>
using namespace std;

struct Node
{
    int x, y;
    int index;

    /**
     * \brief Custom Node << operator, useful for printing out nodes
     */
    friend ostream& operator<<(ostream& o, const Node& node)
    {
        return (o << "Node " << node.index << " is at (" << node.x << ", " << node.y << ")");
    }
};

class ProblemInstance
{

};


#endif //VRP_PROBLEMINSTANCE_CUH
