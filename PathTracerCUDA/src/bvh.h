#pragma once
#include "utility.h"
#include "vector3.h"
#include "hittable.h"
#include "sphere.h"




/// <summary>
/// A bounding volume hierarchy (BVH) is a tree structure on a set of geometric objects.
/// All objects are wrapped in bounding volumes that form the leaf nodes of the tree.
/// Every leaf node contains 1 or 2 objects.
/// These nodes are then grouped as small sets and enclosed withing larger volumes.
/// </summary>
class BVHNode {

public:

	BVHNode() = default;

	BVHNode(std::vector<Sphere>& list, int start, int end, std::vector<BVHNode>& nodes, int* indexNode, int axis) {
		int n = end - start + 1;
		if (n == 1) {
			left = start;
			right = start;
			box = list[left].Box();
			type = 0;
		}
		else if (n == 2) {
			if (BoxCompare(list[start].Box(), list[end].Box(), axis)) {
				left = start;
				right = end;
			}
			else {
				left = end;
				right = start;
			}
			box = MergeBoxes(list[left].Box(), list[right].Box());
			type = 0;
		}
		else {
			// Sort
			auto comparator = 
				  (axis == 0) ? BoxCompareX
				: (axis == 1) ? BoxCompareY
							  : BoxCompareZ;

			std::sort(list.begin() + start, list.begin() + end + 1, comparator);

			int mid = start + (n / 2);
			int lastNodeId = *indexNode;
			*indexNode += 1;
			nodes[lastNodeId] = BVHNode(list, start, mid, nodes, indexNode, (axis + 1) % 3);
			left = lastNodeId;
			lastNodeId = *indexNode;
			*indexNode += 1;
			nodes[lastNodeId] = BVHNode(list, mid + 1, end, nodes, indexNode, (axis + 1) % 3);
			right = lastNodeId;
			box = MergeBoxes(nodes[left].box, nodes[right].box);
			type = axis + 1;
		}
		 
	}

	/// <summary>
	/// Get intersected point between the ray and the BVH node (non-recursive implementation).
	/// </summary>
	/// <param name="r">ray</param>
	/// <param name="t_min">minimal t</param>
	/// <param name="t_max">maximal t</param>
	/// <param name="rec">hit record</param>
	/// <param name="hittables">list of hittable objects</param>
	/// <param name="nodes">list of BVH nodes</param>
	/// <returns>true if ray hit an intersectable</returns>
	__device__ bool Hit(const Ray& r, float t_min, float t_max, HitRecord& rec, Sphere* hittables, BVHNode* nodes) const;


public:
	int type; // Leaf flag + axis
	int left; // Pointer to the left child
	int right; // Pointer to the right child
	AABB box; // Bounding volume of the node
};

struct StackEntry {
	BVHNode* ptr;
	float tmin;
	float tmax;
};

__device__ bool BVHNode::Hit(const Ray& r, float tmin, float tmax, HitRecord& rec, Sphere* hittables, BVHNode* nodes) const {

	//if (!nodes[0].Hit(r, tmin, tmax, rec, hittables, nodes)) {
	//	return false;
	//}
	//tmax = fminf(tmax, rec.t);
	//StackEntry stack[20];
	//int stackId = 1;
	//BVHNode* node = &nodes[0];
	//float saveMaxT = tmax;
	//while (stackId > 0) {
	//	while (node) {
	//		if (node->type == 0) {
	//			// A leaf node
	//			if (node->Hit(r, tmin, tmax, rec, hittables, nodes)) {
	//				tmax = fminf(tmax, rec.t);
	//			}
	//			break;
	//		}
	//		else {
	//			// An internal node
	//			if (node->Hit(r, tmin, tmax, rec, hittables, nodes)) {
	//				BVHNode* nearSubtree;
	//				BVHNode* farSubtree;
	//				if (r.direction[node->type - 1] > 0) {
	//					nearSubtree = &nodes[node->left];
	//					farSubtree = &nodes[node->right];
	//				}
	//				else {
	//					nearSubtree = &nodes[node->right];
	//					farSubtree = &nodes[node->left];
	//				}
	//				if (farSubtree) {
	//					stack[stackId].ptr = farSubtree;
	//					stack[stackId].tmin = tmin;
	//					stack[stackId].tmax = tmax;
	//					stackId++;
	//				}
	//				if (nearSubtree) {
	//					node = nearSubtree;
	//				}
	//				else {
	//					break;
	//				}
	//			}
	//			else {
	//				break;
	//			}
	//		}
	//	}
	//	stackId--;
	//	tmin = stack[stackId].tmin;
	//	tmax = stack[stackId].tmax;
	//	node = stack[stackId].ptr;
	//	if (rec.hit == 1) {
	//		if (tmin > rec.t) {
	//			while (stackId > 0) {
	//				stackId--;
	//				tmin = stack[stackId].tmin;
	//				if (tmin < rec.t) break;
	//			}
	//			node = stack[stackId].ptr;
	//			tmax = stack[stackId].tmax;
	//		}
	//	}
	//}
	//if (rec.hit == 1) {
	//	return true;
	//}
	//else {
	//	return false;
	//}

	BVHNode* stack[20];
	stack[0] = &nodes[0];
	int stackId = 0;
	bool hit = false;
	if (stack[0]->box.Hit(r, tmin, tmax)) {
		while (stackId != -1) {
			BVHNode* currNode = stack[stackId];
			--stackId;
			bool hitLeft = false;
			bool hitRight = false;
			if (currNode->type == 0) {
				Sphere& sLeft = hittables[currNode->left];
				HitRecord tmpRec;
				hitLeft = sLeft.Hit(r, tmin, tmax, tmpRec);
				if (hitLeft) {
					if (hit) {
						if (tmpRec.t < rec.t) {
							rec = tmpRec;
							tmax = rec.t;
						}
					}
					else {
						hit = true;
						rec = tmpRec;
						tmax = rec.t;
					}
				}
				if (currNode->left != currNode->right) {
					Sphere& sRight = hittables[currNode->right];
					hitRight = sRight.Hit(r, tmin, hitLeft ? rec.t : tmax, tmpRec);
				}
				if (hitRight) {
					if (hit) {
						if (tmpRec.t < rec.t) {
							rec = tmpRec;
							tmax = rec.t;
						}
					}
					else {
						hit = true;
						rec = tmpRec;
						tmax = rec.t;
					}
				}
			}
			else {
				if (currNode->box.Hit(r, tmin, tmax)) {
					if ((currNode->type - 1) > 0.0f) {
						stack[++stackId] = &nodes[currNode->left];
						stack[++stackId] = &nodes[currNode->right];
					}
					else {
						stack[++stackId] = &nodes[currNode->right];
						stack[++stackId] = &nodes[currNode->left];
					}
				}
			}
		}
	}
	return hit;
}



